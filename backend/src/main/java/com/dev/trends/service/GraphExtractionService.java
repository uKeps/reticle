package com.dev.trends.service;

import com.dev.trends.model.EdgeRequest;
import com.dev.trends.model.ExtractionResult;
import com.dev.trends.model.NodeRequest;
import com.dev.trends.repository.EdgeRepository;
import com.dev.trends.repository.NodeRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.CallNotPermittedException;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipException;

@Service
public class GraphExtractionService {

    private static final Logger log = LoggerFactory.getLogger(GraphExtractionService.class);

    private static final String HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json";
    private static final String HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/%d.json";
    private static final String DEVTO_URL = "https://dev.to/api/articles?tag=%s&top=7&per_page=12";
    private static final String LOBSTERS_URL = "https://lobste.rs/hottest.json";
    private static final String USER_AGENT = "reticle/1.0 (tech trends aggregator)";

    private static final List<String> DEVTO_TAGS = List.of("ai", "javascript", "rust", "devops", "webdev");

    private static final String STACKOVERFLOW_URL =
            "https://api.stackexchange.com/2.3/questions?order=desc&sort=activity&tagged=%s&site=stackoverflow&pagesize=12&filter=default";
    private static final List<String> STACKOVERFLOW_TAGS =
            List.of("python", "javascript", "typescript", "docker", "kubernetes");

    private static final int MAX_HN_ARTICLES = 60;

    /**
     * How many articles make it into the extraction prompt. Groq's free tier caps
     * at 8000 tokens per minute and counts input + max_tokens as reserved, so the
     * whole request must fit under that ceiling: ~20 tokens per article here gives
     * ~2.7k of input, which combined with max_tokens leaves ~6.7k. We deliberately
     * collect more articles than that — the news feed uses the full list, only the
     * graph curation sees the sample.
     */
    private static final int MAX_PROMPT_ARTICLES = 120;

    /**
     * Output budget of the extraction call. gpt-oss is a reasoning model, so this
     * budget covers the hidden reasoning AND the JSON. Raising it is capped by
     * Groq's 8000 tokens/minute free tier (input + max_tokens are reserved
     * together); the prompt keeps the JSON small instead (node/edge caps, compact
     * output) so it fits.
     */
    private static final int EXTRACTION_MAX_TOKENS = 4000;

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final NodeRepository nodeRepository;
    private final EdgeRepository edgeRepository;
    private final CircuitBreakerRegistry circuitBreakerRegistry;
    /**
     * Used to scope transactions ONLY around persistence (persistNodes/persistEdges).
     * The full pipeline does not run inside a transaction: the ~185 HTTP calls and the
     * synchronous LLM request happen without holding a pool connection (only 3 slots
     * on Render's free tier). Each individual upsert is already atomic via ON CONFLICT;
     * nodes and edges belong to independent aggregates, so we wouldn't need a "global"
     * transaction — we only want each upsert to grab and release the pool quickly.
     */
    private final TransactionTemplate transactionTemplate;

    @Value("${openai.api.key:${GROQ_API_KEY:}}")
    private String llmApiKey;

    @Value("${openai.api.url:https://api.groq.com/openai/v1/chat/completions}")
    private String llmApiUrl;

    @Value("${openai.model:openai/gpt-oss-20b}")
    private String llmModel;

    public GraphExtractionService(
            NodeRepository nodeRepository,
            EdgeRepository edgeRepository,
            ObjectMapper objectMapper,
            HttpClient httpClient,
            PlatformTransactionManager transactionManager,
            CircuitBreakerRegistry circuitBreakerRegistry) {
        this.nodeRepository = nodeRepository;
        this.edgeRepository = edgeRepository;
        this.objectMapper = objectMapper;
        this.httpClient = httpClient;
        this.circuitBreakerRegistry = circuitBreakerRegistry;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
    }

    /** Artigo coletado de qualquer plataforma. */
    public record Article(String id, String title, String url, String discussionUrl, String platform, Instant publishedAt) {}

    /**
     * Maximum age of articles in the news feed and the LLM prompt. Items older
     * than this are dropped at collection time — protects against Lobsters "hottest"
     * (all-time) and StackOverflow "activity" (old questions with a recent comment)
     * pulling in years-old content, which defeats the purpose of a "what's trending
     * now" site.
     */
    private static final Duration MAX_ARTICLE_AGE = Duration.ofDays(30);

    /**
     * Main pipeline: collects articles from multiple sources -> extracts the graph
     * via the LLM -> persists. Intentionally NOT {@code @Transactional}: the
     * pipeline includes ~185 HTTP requests and a synchronous LLM call (up to 60s).
     * Holding a transaction across that would monopolize a Hikari connection
     * (3-slot pool) and blow past the limit on any pairing. Persistence is wrapped
     * in a short transaction inside persistNodes/persistEdges.
     */
    public ExtractionResult runIngestionPipeline() {
        log.info("Starting multi-source ingestion pipeline...");

        List<Article> articles = fetchAllArticles();
        if (articles.isEmpty()) {
            log.warn("No articles collected. Aborting pipeline.");
            return ExtractionResult.empty();
        }
        log.info("Coletados {} artigos de {} fontes.", articles.size(),
                articles.stream().map(Article::platform).distinct().count());

        ExtractionResult extracted = callLlmForExtraction(articles);
        log.info("LLM extracted {} nodes and {} edges.", extracted.nodes().size(), extracted.edges().size());

        int nodesCreated = persistNodes(extracted.nodes(), articles);
        int edgesCreated = persistEdges(extracted.edges(), articles);

        log.info("Pipeline finished. Nodes: {}, Edges: {}", nodesCreated, edgesCreated);
        return extracted;
    }

    // =========================================================
    // PHASE 1: Multi-source collection
    // =========================================================

    private List<Article> fetchAllArticles() {
        List<Article> all = new ArrayList<>();
        all.addAll(fetchHackerNewsArticles());
        all.addAll(fetchDevToArticles());
        all.addAll(fetchLobstersArticles());
        all.addAll(fetchStackOverflowArticles());
        return all;
    }

    private List<Article> fetchHackerNewsArticles() {
        return runWithBreaker("hackernews", List.of(), this::doFetchHackerNewsArticles);
    }

    private List<Article> doFetchHackerNewsArticles() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(HN_TOP_STORIES_URL))
                    .timeout(Duration.ofSeconds(10))
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return List.of();

            JsonNode idsNode = objectMapper.readTree(resp.body());
            List<Long> storyIds = new ArrayList<>();
            for (int i = 0; i < Math.min(MAX_HN_ARTICLES, idsNode.size()); i++) {
                storyIds.add(idsNode.get(i).asLong());
            }

            return storyIds.stream()
                    .map(id -> CompletableFuture.supplyAsync(() -> fetchHnItem(id),
                            httpClient.executor().orElse(Runnable::run)))
                    .map(CompletableFuture::join)
                    .filter(a -> a != null)
                    .filter(a -> !isOlderThanCeiling(a.publishedAt()))
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Erro ao coletar Hacker News: {}", e.getMessage());
            return List.of();
        }
    }

    private Article fetchHnItem(long itemId) {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(HN_ITEM_URL.formatted(itemId)))
                    .timeout(Duration.ofSeconds(8))
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                JsonNode item = objectMapper.readTree(resp.body());
                if (item.has("title")) {
                    String title = item.get("title").asText();
                    String url = item.has("url") ? item.get("url").asText()
                            : "https://news.ycombinator.com/item?id=" + itemId;
                    String discussion = "https://news.ycombinator.com/item?id=" + itemId;
                    // HN's "time" field is epoch seconds. Without it (item removed, etc.)
                    // we leave it null and the age filter on the layer above still runs.
                    Instant publishedAt = item.has("time") && item.get("time").isNumber()
                            ? Instant.ofEpochSecond(item.get("time").asLong())
                            : null;
                    return new Article(String.valueOf(itemId), title, url, discussion, "hackernews", publishedAt);
                }
            }
        } catch (Exception e) {
            log.debug("Erro ao buscar item HN {}: {}", itemId, e.getMessage());
        }
        return null;
    }

    /**
     * True if `publishedAt` is older than MAX_ARTICLE_AGE. When the date came back
     * null (field missing at the source, broken parse, etc.) we accept the article —
     * the hard ceiling at the fetch* entry still protects against obvious garbage.
     */
    private boolean isOlderThanCeiling(Instant publishedAt) {
        if (publishedAt == null) return false;
        return publishedAt.isBefore(Instant.now().minus(MAX_ARTICLE_AGE));
    }

    private List<Article> fetchDevToArticles() {
        return runWithBreaker("devto", List.of(), this::doFetchDevToArticles);
    }

    private List<Article> doFetchDevToArticles() {
        List<Article> articles = new ArrayList<>();
        for (String tag : DEVTO_TAGS) {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(DEVTO_URL.formatted(tag)))
                        .timeout(Duration.ofSeconds(10))
                        .header("User-Agent", USER_AGENT)
                        .GET()
                        .build();

                HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() != 200) continue;

                JsonNode items = objectMapper.readTree(resp.body());
                if (!items.isArray()) continue;

                for (JsonNode item : items) {
                    String title = item.path("title").asText("");
                    if (title.isBlank()) continue;

                    String url = item.path("url").asText("");
                    String id = String.valueOf(item.path("id").asLong());
                    // Dev.to returns ISO 8601 in "published_at"; it can be missing for
                    // legacy drafts. Tolerant parse - failure becomes null.
                    Instant publishedAt = parseInstantOrNull(item.path("published_at").asText(""));
                    if (isOlderThanCeiling(publishedAt)) continue;
                    articles.add(new Article(id, title, url, url, "devto", publishedAt));
                }
            } catch (Exception e) {
                log.debug("Erro ao coletar Dev.to tag {}: {}", tag, e.getMessage());
            }
        }
        log.info("Dev.to: {} artigos coletados.", articles.size());
        return articles;
    }

    private List<Article> fetchLobstersArticles() {
        return runWithBreaker("lobsters", List.of(), this::doFetchLobstersArticles);
    }

    private List<Article> doFetchLobstersArticles() {
        List<Article> articles = new ArrayList<>();
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(LOBSTERS_URL))
                    .timeout(Duration.ofSeconds(10))
                    .header("User-Agent", USER_AGENT)
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return List.of();

            JsonNode items = objectMapper.readTree(resp.body());
            if (!items.isArray()) return List.of();

            for (JsonNode item : items) {
                String title = item.path("title").asText("");
                if (title.isBlank()) continue;

                String url = item.path("url").asText("");
                String commentsUrl = item.path("comments_url").asText(url);
                String id = item.path("short_id").asText("");
                // /hottest.json is "all-time hottest", so without this filter the home
                // page fills with items from years ago that went viral once and never came back.
                Instant publishedAt = parseInstantOrNull(item.path("created_at").asText(""));
                if (isOlderThanCeiling(publishedAt)) continue;
                articles.add(new Article(id, title, url, commentsUrl, "lobsters", publishedAt));
            }
        } catch (Exception e) {
            log.debug("Erro ao coletar Lobsters: {}", e.getMessage());
        }
        log.info("Lobsters: {} artigos coletados.", articles.size());
        return articles;
    }

    private List<Article> fetchStackOverflowArticles() {
        return runWithBreaker("stackoverflow", List.of(), this::doFetchStackOverflowArticles);
    }

    private List<Article> doFetchStackOverflowArticles() {
        List<Article> articles = new ArrayList<>();
        for (String tag : STACKOVERFLOW_TAGS) {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(STACKOVERFLOW_URL.formatted(tag)))
                        .timeout(Duration.ofSeconds(10))
                        .header("User-Agent", USER_AGENT)
                        .GET()
                        .build();

                HttpResponse<byte[]> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofByteArray());
                if (resp.statusCode() != 200) continue;

                String body = decodeStackExchangeBody(resp.body());
                JsonNode items = objectMapper.readTree(body).path("items");
                if (!items.isArray()) continue;

                for (JsonNode item : items) {
                    String title = item.path("title").asText("");
                    if (title.isBlank()) continue;

                    String link = item.path("link").asText("");
                    String id = String.valueOf(item.path("question_id").asLong());
                    // SO devolve epoch em segundos em "creation_date". Com sort=activity, sem
                    // without this filter we get years-old questions that just received a comment.
                    Instant publishedAt = item.has("creation_date") && item.get("creation_date").isNumber()
                            ? Instant.ofEpochSecond(item.get("creation_date").asLong())
                            : null;
                    if (isOlderThanCeiling(publishedAt)) continue;
                    articles.add(new Article(id, title, link, link, "stackoverflow", publishedAt));
                }
            } catch (Exception e) {
                log.debug("Erro ao coletar StackOverflow tag {}: {}", tag, e.getMessage());
            }
        }
        log.info("StackOverflow: {} artigos coletados.", articles.size());
        return articles;
    }

    /**
     * Tenta parsear uma string ISO 8601 (ou vazia) em Instant. Retorna null em qualquer falha —
     * usada para campos de data opcionais nas respostas das fontes.
     */
    private static Instant parseInstantOrNull(String text) {
        if (text == null || text.isBlank()) return null;
        try {
            return Instant.parse(text);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * A API do Stack Exchange sempre comprime a resposta em gzip, mesmo sem pedir — precisa
     * descomprimir manualmente antes de fazer o parse do JSON.
     */
    private String decodeStackExchangeBody(byte[] raw) throws IOException {
        try (var gzipStream = new GZIPInputStream(new ByteArrayInputStream(raw))) {
            return new String(gzipStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (ZipException e) {
            return new String(raw, StandardCharsets.UTF_8);
        }
    }

    // =========================================================
    // PHASE 2: LLM extraction
    // =========================================================

    private static final List<String> BLACKLIST = NodeRepository.BLACKLIST;

    /** Result of a single LLM call. `content` is null on failure; `error` describes
     *  the failure (null on success). Equivalent to a Result<String, String> to
     *  avoid a mutable field shared across requests. */
    private record LlmCallResult(String content, String error) {
        boolean failed() { return content == null; }
    }

    /**
     * Calls the LLM and returns the cleaned-up content. On failure, returns the
     * reason in the `error` field (per call) — never writes to an instance field,
     * so concurrent requests don't clobber each other.
     */
    private LlmCallResult requestLlmContent(String model, String systemPrompt, String userMessage) {
        return runWithBreaker("llm", new LlmCallResult(null, "LLM breaker open: " + model),
                () -> doRequestLlmContent(model, systemPrompt, userMessage));
    }

    private LlmCallResult doRequestLlmContent(String model, String systemPrompt, String userMessage) {
        try {
            Map<String, Object> requestBody = Map.of(
                    "model", model,
                    "temperature", 0.1,
                    // Output + input must fit inside Groq's 8000 tokens/minute free tier,
                    // which counts both against the same limit (413 rate_limit_exceeded when
                    // exceeded). With ~2.7k of input this keeps the request at ~6.7k.
                    "max_tokens", EXTRACTION_MAX_TOKENS,
                    "reasoning_effort", "low",
                    // JSON mode: the content is a bare JSON object, no prose or markdown fences.
                    "response_format", Map.of("type", "json_object"),
                    "messages", List.of(
                            Map.of("role", "system", "content", systemPrompt),
                            Map.of("role", "user", "content", userMessage)
                    )
            );

            HttpRequest llmReq = HttpRequest.newBuilder()
                    .uri(URI.create(llmApiUrl))
                    .timeout(Duration.ofSeconds(60))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + llmApiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(requestBody)))
                    .build();

            HttpResponse<String> llmResp = httpClient.send(llmReq, HttpResponse.BodyHandlers.ofString());

            if (llmResp.statusCode() != 200) {
                log.error("LLM API ({}) status {}. Body: {}", model, llmResp.statusCode(), llmResp.body());
                String body = llmResp.body();
                String error = "HTTP " + llmResp.statusCode() + " em " + model + ": "
                        + (body.length() > 400 ? body.substring(0, 400) : body);
                return new LlmCallResult(null, error);
            }

            JsonNode root = objectMapper.readTree(llmResp.body());
            JsonNode choice = root.path("choices").get(0);
            String content = choice.path("message").path("content").asText("");

            // The output budget ran out (in the reasoning or mid-JSON): whatever came
            // back is cut off. Report it as such instead of as a generic parse error.
            if ("length".equals(choice.path("finish_reason").asText())) {
                log.error("LLM {} hit max_tokens={} ({} chars of content)", model,
                        EXTRACTION_MAX_TOKENS, content.length());
                return new LlmCallResult(null, "LLM output truncated at max_tokens="
                        + EXTRACTION_MAX_TOKENS + " (finish_reason=length) from " + model);
            }

            content = content.replaceAll("```json", "").replaceAll("```", "").trim();
            log.info("Resposta bruta do LLM {} ({} chars): {}", model, content.length(),
                    content.length() > 200 ? content.substring(0, 200) + "..." : content);
            return new LlmCallResult(content, null);

        } catch (Exception e) {
            log.error("Erro na chamada ao LLM ({}): {}", model, e.getMessage(), e);
            return new LlmCallResult(null, model + ": " + e);
        }
    }

    /**
     * Evenly-spaced sample of the collected list, so the prompt fits the tokens-per-minute
     * limit without losing the source mix - sources arrive concatenated (HN, Dev.to,
     * Lobsters, SO), so taking the first N would leave the trailing sources out of curation.
     */
    List<Article> sampleForPrompt(List<Article> articles) {
        if (articles.size() <= MAX_PROMPT_ARTICLES) return articles;

        List<Article> sample = new ArrayList<>(MAX_PROMPT_ARTICLES);
        double step = (double) articles.size() / MAX_PROMPT_ARTICLES;
        for (int i = 0; i < MAX_PROMPT_ARTICLES; i++) {
            sample.add(articles.get((int) (i * step)));
        }
        return sample;
    }

    ExtractionResult callLlmForExtraction(List<Article> articles) {
        if (llmApiKey == null || llmApiKey.isBlank()) {
            log.warn("LLM API key not configured. Falling back to keyword extraction.");
            return extractByKeyword(articles, "GROQ_API_KEY missing from service configuration");
        }

        // Without the links: the source for each node is resolved locally in parseLlmResponse,
        // so sending URLs (and asking the model to echo them back) would just burn input/output
        // tokens for nothing.
        String articlesBlock = sampleForPrompt(articles).stream()
                .map(a -> "- [" + a.platform().toUpperCase() + "] " + a.title())
                .collect(Collectors.joining("\n"));

        log.debug("Articles sent to LLM:\n{}", articlesBlock);

        // The system prompt and user message below are intentionally in Portuguese:
        // they instruct the LLM to play the role of a senior technical curator and to
        // produce English node labels (rule #2). Do not translate these strings —
        // changing the prompt language would change the model's behavior.
        String systemPrompt = """
                Você é um Curador Técnico Sênior especializado em Engenharia de Software e IA.
                Analise a lista de matérias/discussões de Hacker News, Reddit, Dev.to e Lobsters.
                Extraia ferramentas, linguagens, frameworks e modelos com ALTO VALOR DE APRENDIZADO.

                REGRAS OBRIGATÓRIAS:
                1. NÃO INCLUA termos genéricos (Linux, Mac, Windows, Software, Hardware, Web, Computer, Article, PDF).
                2. Use o label exatamente como ele aparece escrito no título da matéria.
                3. A lista contém matérias NÃO TÉCNICAS (história, política, saúde, drogas,
                   curiosidades, cultura pop) — o Hacker News publica isso na home. IGNORE essas
                   matérias por completo. Só entram no grafo ferramentas, linguagens, frameworks,
                   modelos e conceitos de engenharia de software e IA. Na dúvida, deixe de fora.
                4. No máximo 40 nós e 40 arestas — fique só com os mais relevantes.
                5. JSON compacto, em uma única linha, sem indentação nem quebras de linha.

                Responda APENAS com JSON válido no formato:
                {
                  "nodes": [
                    {"label": "LangGraph", "category": "Framework"}
                  ],
                  "edges": [
                    {"source": "LangGraph", "target": "LangChain", "relation": "PART_OF"}
                  ]
                }
                """;

        String userMessage = "Analise estas matérias e extraia o grafo de conhecimento:\n\n" + articlesBlock;

        LlmCallResult llmCall = requestLlmContent(llmModel, systemPrompt, userMessage);

        if (llmCall.failed() || !llmCall.content().trim().startsWith("{")) {
            log.error("Model {} did not return valid JSON. Falling back to keyword extraction. Response: '{}'",
                    llmModel, llmCall.content());
            String error = llmCall.error() != null
                    ? llmCall.error()
                    : "response without JSON from " + llmModel;
            return extractByKeyword(articles, error);
        }

        // An unparseable answer still gets the keyword fallback, so the run never
        // ends with an empty graph just because the JSON was malformed.
        ExtractionResult parsed = parseLlmResponse(llmCall.content(), articles);
        if (parsed.llmError() != null) {
            return extractByKeyword(articles, parsed.llmError());
        }
        return parsed;
    }

    /**
     * Cache of compiled Patterns per word. {@link #containsWord} runs on the hot
     * path (once per node per article in {@code parseLlmResponse}, plus in
     * {@code persistNodeArticles} and {@code findLiveSource}); recompiling the regex
     * on every call became the dominant cost of that step. Caching is transparent:
     * the resulting matcher is byte-for-byte identical to a fresh one.
     */
    private static final ConcurrentHashMap<String, Pattern> WORD_PATTERNS = new ConcurrentHashMap<>();

    /**
     * Checks whether `word` appears in `text` as a whole word (not as a substring of
     * a larger word). Prevents "Java" from matching inside "JavaScript" and vice versa.
     */
    private boolean containsWord(String text, String word) {
        if (text == null || word == null || word.isBlank()) return false;
        Pattern pattern = WORD_PATTERNS.computeIfAbsent(word,
                w -> Pattern.compile("\\b" + Pattern.quote(w) + "\\b", Pattern.CASE_INSENSITIVE));
        return pattern.matcher(text).find();
    }

    /**
     * Pre-computed index of normalized label -> original article. Allows
     * {@link #findBestMatchingArticle} to run in O(1) instead of scanning the whole
     * article list for every node/edge processed. Preserves "first match wins" by
     * iterating the articles in their original order when building the index.
     */
    private Map<String, Article> indexArticlesByLabel(List<Article> articles) {
        Map<String, Article> index = new HashMap<>();
        if (articles == null) return index;
        for (Article article : articles) {
            if (article == null || article.title() == null) continue;
            String[] words = article.title().split("\\s+");
            for (String word : words) {
                String trimmed = word.toLowerCase();
                index.putIfAbsent(trimmed, article);
            }
        }
        return index;
    }

    /**
     * Indexed variant of {@link #findBestMatchingArticle}. Looks up in the
     * pre-computed map (instead of iterating the list) — semantically equivalent
     * when the index was built from the same list.
     */
    private Article findBestMatchingArticleFromIndex(String label, Map<String, Article> index) {
        if (label == null || label.isBlank() || index.isEmpty()) return null;
        String[] words = label.split("\\s+");
        for (String word : words) {
            Article match = index.get(word.toLowerCase());
            if (match != null) return match;
        }
        return null;
    }

    /**
     * Finds the original article whose title mentions the extracted label.
     * Used to reliably assign a real source, without depending on the LLM to reproduce URLs.
     */
    private Article findBestMatchingArticle(String label, List<Article> articles) {
        return articles.stream()
                .filter(a -> containsWord(a.title(), label))
                .findFirst()
                .orElse(null);
    }

    /**
     * Looks up a real source on demand for a topic that has no stored sourceUrl, using
     * the public Hacker News search API (Algolia). Called on demand, when the user
     * clicks — it doesn't depend on the topic happening to appear in an ingestion batch.
     */
    public Article findLiveSource(String label) {
        return runWithBreaker("hackernews", (Article) null, () -> doFindLiveSource(label));
    }

    private Article doFindLiveSource(String label) {
        try {
            String query = URLEncoder.encode(label, StandardCharsets.UTF_8);
            // hitsPerPage > 1 because Algolia's relevance isn't whole-word-aware: a search
            // for "Java" usually returns "JavaScript" results at the top. We filter below.
            String url = "https://hn.algolia.com/api/v1/search?query=" + query
                    + "&tags=story&hitsPerPage=20";

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(8))
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) return null;

            JsonNode hits = objectMapper.readTree(resp.body()).path("hits");
            if (!hits.isArray() || hits.isEmpty()) return null;

            for (JsonNode hit : hits) {
                String title = hit.path("title").asText("");
                if (title.isBlank() || !containsWord(title, label)) continue;

                String objectId = hit.path("objectID").asText("");
                String discussion = "https://news.ycombinator.com/item?id=" + objectId;
                String articleUrl = hit.path("url").asText(discussion);
                // Algolia returns "created_at" as ISO 8601.
                Instant publishedAt = parseInstantOrNull(hit.path("created_at").asText(""));

                return new Article(objectId, title, articleUrl, discussion, "hackernews", publishedAt);
            }
            return null;
        } catch (Exception e) {
            log.debug("Error fetching live source for '{}': {}", label, e.getMessage());
            return null;
        }
    }

    private ExtractionResult parseLlmResponse(String content, List<Article> articles) {
        try {
            JsonNode graphJson = objectMapper.readTree(content);

            Map<String, Article> articleIndex = indexArticlesByLabel(articles);

            List<NodeRequest> nodes = StreamSupport.stream(
                            graphJson.path("nodes").spliterator(), false)
                    .map(n -> {
                        String label = n.path("label").asText("Unknown").trim();
                        String category = n.path("category").asText("Technology").trim();

                        // The source comes from the collected batch, not from the LLM: the URL
                        // it returned was discarded when it didn't match a real article
                        // (hallucination), and the rest of the fields already came from here.
                        // Without a source, the frontend looks one up on demand.
                        Article match = findBestMatchingArticleFromIndex(label, articleIndex);
                        String sourceUrl = match != null ? match.discussionUrl() : "";
                        String sourceTitle = match != null ? match.title() : "Discussion in the dev community";
                        String sourcePlatform = match != null ? match.platform() : "web";

                        return new NodeRequest(label, category, null, sourceUrl, sourceTitle, sourcePlatform);
                    })
                    .filter(n -> !n.label().isBlank())
                    .filter(n -> !isBlacklisted(n.label()))
                    .toList();

            List<EdgeRequest> edges = StreamSupport.stream(
                            graphJson.path("edges").spliterator(), false)
                    .map(e -> new EdgeRequest(
                            e.path("source").asText().trim(),
                            e.path("target").asText().trim(),
                            e.path("relation").asText("RELATED_TO").trim()))
                    .filter(e -> !e.source().isBlank() && !e.target().isBlank())
                    .filter(e -> !isBlacklisted(e.source()) && !isBlacklisted(e.target()))
                    .toList();

            return new ExtractionResult(nodes, edges, null);

        } catch (Exception e) {
            log.error("Failed to parse LLM response: {}", e.getMessage(), e);
            return ExtractionResult.withError("failed to parse LLM JSON: " + e.getMessage());
        }
    }

    private static final String SUMMARY_PROMPT_EN = """
            Explain ONE specific technology technically, in 2-3 sentences, in English.
            Say what it IS and what it DOES — do not talk about "trends" or "recent discussions".

            BAD (do not do this): "React is a rising technology in the dev ecosystem, with recent discussions in the developer bubble."

            GOOD: "React is a JavaScript library for building declarative component-based interfaces, using a virtual DOM to optimize re-renders."
            """;

    // The two summary prompts below are intentionally per-language user-facing prompts
    // sent to the LLM. They instruct the model to produce summaries in the requested
    // language and must not be translated — changing the prompt language would change
    // the model's behavior.
    private static final String SUMMARY_PROMPT_PT = """
            Explique tecnicamente UMA tecnologia específica em 2-3 frases em português.
            Diga o que ela É e o que ela FAZ — não fale sobre "tendências" ou "discussões recentes".

            RUIM (não faça isso): "React é uma tecnologia em ascensão no ecossistema dev, com discussões recentes na bolha de desenvolvimento."

            BOM: "React é uma biblioteca JavaScript para construir interfaces declarativas baseadas em componentes, usando um DOM virtual para otimizar re-renderizações."
            """;

    /**
     * Generates a specific, technical summary for ONE technology on demand, in the
     * requested language ("pt" for Portuguese, anything else falls back to English,
     * which is the UI default).
     */
    public String generateTopicSummary(String label, String category, String sourceTitle, String sourceUrl, String lang) {
        if (llmApiKey == null || llmApiKey.isBlank()) {
            log.warn("LLM API key not configured. Cannot generate on-demand summary.");
            return null;
        }

        boolean pt = "pt".equals(lang);
        String system = pt ? SUMMARY_PROMPT_PT : SUMMARY_PROMPT_EN;
        String user = (pt ? "Tecnologia: %s\nCategoria: %s\nContexto (artigo que a mencionou): \"%s\" (%s)"
                          : "Technology: %s\nCategory: %s\nContext (article that mentioned it): \"%s\" (%s)")
                .formatted(label, category, sourceTitle != null ? sourceTitle : "", sourceUrl != null ? sourceUrl : "");

        try {
            Map<String, Object> requestBody = Map.of(
                    "model", llmModel,
                    "temperature", 0.4,
                    "max_tokens", 400,
                    "reasoning_effort", "low",
                    "messages", List.of(
                            Map.of("role", "system", "content", system),
                            Map.of("role", "user", "content", user)
                    )
            );

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(llmApiUrl))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + llmApiKey)
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(requestBody)))
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());

            if (resp.statusCode() == 200) {
                JsonNode root = objectMapper.readTree(resp.body());
                String content = root.path("choices").get(0).path("message").path("content").asText("");
                content = content.replaceAll("```markdown", "").replaceAll("```", "").trim();
                if (!content.isBlank()) {
                    return content;
                }
            } else {
                log.error("Error generating summary for {}: status {}", label, resp.statusCode());
            }
        } catch (Exception e) {
            log.error("Error generating summary for {}: {}", label, e.getMessage());
        }
        return null;
    }

    private boolean isBlacklisted(String label) {
        if (label == null || label.isBlank()) return true;
        String lower = label.toLowerCase().trim();
        return BLACKLIST.stream().anyMatch(b -> lower.equals(b) || lower.startsWith(b + " ") || lower.endsWith(" " + b));
    }

    private ExtractionResult extractByKeyword(List<Article> articles, String llmError) {
        List<String> titles = articles.stream().map(Article::title).toList();
        List<String> techKeywords = List.of(
                "AI", "LLM", "GPT", "Claude", "Gemini", "Llama", "Python", "Java", "JavaScript", "Rust",
                "TypeScript", "React", "Next.js", "Kubernetes", "Docker", "AWS", "GCP",
                "PostgreSQL", "Redis", "GraphQL", "WebAssembly", "Deno", "Bun", "Vite",
                "LangChain", "LangGraph", "OpenAI", "Anthropic", "Groq", "Mistral",
                "RAG", "Vector Database", "Embedding", "Agent", "MCP", "Spring Boot"
        );

        Map<String, Article> articleIndex = indexArticlesByLabel(articles);

        Map<String, Long> mentionCounts = techKeywords.stream()
                .filter(kw -> titles.stream().anyMatch(t -> containsWord(t, kw)))
                .collect(Collectors.groupingBy(kw -> kw, Collectors.counting()));

        List<NodeRequest> nodes = mentionCounts.entrySet().stream()
                .map(e -> {
                    Article match = findBestMatchingArticleFromIndex(e.getKey(), articleIndex);
                    if (match != null) {
                        return new NodeRequest(e.getKey(), categorize(e.getKey()), null,
                                match.discussionUrl(), match.title(), match.platform());
                    }
                    return new NodeRequest(e.getKey(), categorize(e.getKey()));
                })
                .toList();

        List<EdgeRequest> edges = new ArrayList<>();
        List<String> mentionedTerms = new ArrayList<>(mentionCounts.keySet());
        for (int i = 0; i < Math.min(mentionedTerms.size() - 1, 15); i++) {
            edges.add(new EdgeRequest(mentionedTerms.get(i), mentionedTerms.get(i + 1), "RELATED_TO"));
        }

        return new ExtractionResult(nodes, edges, llmError);
    }

    private String categorize(String term) {
        return switch (term.toLowerCase()) {
            case "python", "java", "javascript", "rust", "typescript", "deno", "bun" -> "Language";
            case "react", "next.js", "vite", "spring boot", "langchain", "langgraph" -> "Framework";
            case "docker", "kubernetes", "redis", "postgresql", "graphql" -> "Tool";
            case "aws", "gcp", "vercel", "supabase" -> "Platform";
            case "openai", "anthropic", "groq", "mistral" -> "Company";
            case "gpt", "claude", "gemini", "llama" -> "Model";
            default -> "Concept";
        };
    }

    // =========================================================
    // PHASE 3: Persistence
    // =========================================================

    private int persistNodes(List<NodeRequest> nodes, List<Article> articles) {
        return transactionTemplate.execute(status -> {
            int count = 0;
            for (NodeRequest node : nodes) {
                try {
                    UUID nodeId = nodeRepository.upsertNode(node.label(), node.category(), node.summary(),
                            node.sourceUrl(), node.sourceTitle(), node.sourcePlatform());
                    persistNodeArticles(nodeId, node.label(), articles);
                    count++;
                } catch (Exception e) {
                    log.warn("Failed to persist node '{}': {}", node.label(), e.getMessage());
                }
            }
            return count;
        });
    }

    /**
     * Links to the topic every article from the current batch whose title mentions it
     * (not just the first one), to feed the news feed. Capped at 20 per topic per
     * ingestion round.
     */
    private void persistNodeArticles(UUID nodeId, String label, List<Article> articles) {
        if (nodeId == null) return;
        articles.stream()
                .filter(a -> containsWord(a.title(), label))
                .limit(20)
                .forEach(a -> {
                    try {
                        nodeRepository.insertArticle(nodeId, a.title(), a.discussionUrl(), a.platform(), a.publishedAt());
                    } catch (Exception e) {
                        log.debug("Falha ao persistir artigo para '{}': {}", label, e.getMessage());
                    }
                });
    }

    private int persistEdges(List<EdgeRequest> edges, List<Article> articles) {
        return transactionTemplate.execute(status -> {
            // Collect every label referenced by the edges and resolve existing IDs in a
            // single batched query. Previously this was 2 SELECTs per edge - for 50 edges
            // that's 100 Postgres round-trips; now it's 1.
            Set<String> referencedLabels = new HashSet<>();
            for (EdgeRequest edge : edges) {
                if (edge.source() != null && !edge.source().isBlank()) referencedLabels.add(edge.source());
                if (edge.target() != null && !edge.target().isBlank()) referencedLabels.add(edge.target());
            }
            Map<String, UUID> existingIds = nodeRepository.findIdsByLabels(referencedLabels);

            int count = 0;
            for (EdgeRequest edge : edges) {
                try {
                    UUID sourceId = lookupOrCreate(edge.source(), existingIds, articles);
                    UUID targetId = lookupOrCreate(edge.target(), existingIds, articles);

                    if (sourceId != null && targetId != null && !sourceId.equals(targetId)) {
                        edgeRepository.upsertEdge(sourceId, targetId, edge.relation());
                        count++;
                    }
                } catch (Exception e) {
                    log.warn("Failed to persist edge '{}->{}': {}", edge.source(), edge.target(), e.getMessage());
                }
            }
            return count;
        });
    }

    /**
     * Resolves the UUID for a label from the pre-loaded map; if absent, creates an
     * orphan node and populates the map so subsequent calls with the same label
     * don't repeat the work.
     */
    private UUID lookupOrCreate(String label, Map<String, UUID> cache, List<Article> articles) {
        if (label == null || label.isBlank()) return null;
        String key = label.toLowerCase();
        UUID existing = cache.get(key);
        if (existing != null) return existing;

        UUID created = createOrphanNode(label, articles);
        if (created != null) {
            cache.put(key, created);
        }
        return created;
    }

    /**
     * Creates a node referenced only by an edge (the LLM didn't include it in `nodes`).
     * Tries to find a real source in the already-collected article batch; leaves the
     * summary blank (instead of a fixed generic text) so the frontend still triggers
     * the on-demand source lookup afterwards.
     */
    private UUID createOrphanNode(String label, List<Article> articles) {
        Article match = findBestMatchingArticle(label, articles);
        String sourceUrl = match != null ? match.discussionUrl() : null;
        String sourceTitle = match != null ? match.title() : null;
        String sourcePlatform = match != null ? match.platform() : null;
        UUID nodeId = nodeRepository.upsertNode(label, "Concept", null, sourceUrl, sourceTitle, sourcePlatform);
        persistNodeArticles(nodeId, label, articles);
        return nodeId;
    }

    // =========================================================
    // Circuit breaker helper.
    //
    // Programmatic API used instead of @CircuitBreaker annotations because
    // the fetch methods are private and called from the same class — Spring's
    // AOP proxy does not intercept internal calls, so annotations on those
    // methods would be silently ignored.
    // =========================================================

    /**
     * Runs the given supplier through the named circuit breaker. Any exception
     * (network, timeout, parse) OR a CallNotPermittedException from an open
     * breaker resolves to the supplied fallback so the pipeline degrades
     * gracefully instead of taking the whole run down.
     */
    private <T> T runWithBreaker(String name, T fallback, java.util.function.Supplier<T> action) {
        CircuitBreaker breaker = circuitBreakerRegistry.circuitBreaker(name);
        try {
            return breaker.executeSupplier(action);
        } catch (CallNotPermittedException e) {
            log.warn("{} breaker open. Falling back.", name);
            return fallback;
        } catch (Exception e) {
            log.warn("{} call failed: {}", name, e.getMessage());
            return fallback;
        }
    }
}
