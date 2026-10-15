package com.dev.trends.service;

import com.dev.trends.model.ExtractionResult;
import com.dev.trends.model.NodeRequest;
import com.dev.trends.service.GraphExtractionService.Article;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.transaction.PlatformTransactionManager;

import java.io.ByteArrayOutputStream;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * The scheduled ingest workflow fails whenever the extraction degrades. Most of those
 * failures were gpt-oss running out of max_tokens mid-JSON: the run then persisted
 * zero nodes and reported an opaque Jackson error. These tests pin the contract:
 * truncation is named as such, and any unusable answer still gets the keyword fallback.
 */
class GraphExtractionLlmFallbackTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final HttpClient httpClient = mock(HttpClient.class);
    private final GraphExtractionService service = buildService(httpClient);

    private static GraphExtractionService buildService(HttpClient httpClient) {
        GraphExtractionService service = new GraphExtractionService(null, null, MAPPER, httpClient,
                mock(PlatformTransactionManager.class), CircuitBreakerRegistry.ofDefaults());
        ReflectionTestUtils.setField(service, "llmApiKey", "test-key");
        ReflectionTestUtils.setField(service, "llmApiUrl", "http://llm.test/v1/chat/completions");
        ReflectionTestUtils.setField(service, "llmModel", "openai/gpt-oss-20b");
        return service;
    }

    private static List<Article> articles() {
        return List.of(
                new Article("1", "Rust 2.0 released", "https://x/1", "https://x/1", "hackernews", Instant.now()),
                new Article("2", "Building agents with LangGraph", "https://x/2", "https://x/2", "devto", Instant.now()));
    }

    private void llmAnswers(String content, String finishReason) throws Exception {
        String body = MAPPER.writeValueAsString(Map.of("choices", List.of(Map.of(
                "message", Map.of("role", "assistant", "content", content),
                "finish_reason", finishReason))));
        @SuppressWarnings("unchecked")
        HttpResponse<String> response = mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(200);
        when(response.body()).thenReturn(body);
        doReturn(response).when(httpClient).send(any(), any());
    }

    private static List<String> labels(ExtractionResult result) {
        return result.nodes().stream().map(NodeRequest::label).toList();
    }

    @Test
    void parsesAWellFormedAnswer() throws Exception {
        llmAnswers("{\"nodes\":[{\"label\":\"Rust\",\"category\":\"Language\"}],"
                + "\"edges\":[{\"source\":\"LangGraph\",\"target\":\"LangChain\",\"relation\":\"PART_OF\"}]}", "stop");

        ExtractionResult result = service.callLlmForExtraction(articles());

        assertThat(result.llmError()).isNull();
        assertThat(labels(result)).containsExactly("Rust");
        assertThat(result.edges()).hasSize(1);
    }

    @Test
    void reportsTruncationAndFallsBackToKeywords() throws Exception {
        llmAnswers("{\"nodes\":[{\"label\":\"Rust\",\"category\":\"Lang", "length");

        ExtractionResult result = service.callLlmForExtraction(articles());

        assertThat(result.llmError()).contains("truncated").contains("finish_reason=length");
        assertThat(labels(result)).contains("Rust", "LangGraph");
    }

    @Test
    void reportsTruncationEvenWhenTheReasoningAteTheWholeBudget() throws Exception {
        llmAnswers("", "length");

        ExtractionResult result = service.callLlmForExtraction(articles());

        assertThat(result.llmError()).contains("truncated");
        assertThat(result.nodes()).isNotEmpty();
    }

    @Test
    void unparseableJsonFallsBackToKeywordsInsteadOfAnEmptyGraph() throws Exception {
        llmAnswers("{\"nodes\":[{\"label\":\"Rust\",", "stop");

        ExtractionResult result = service.callLlmForExtraction(articles());

        assertThat(result.llmError()).startsWith("failed to parse LLM JSON");
        assertThat(labels(result)).contains("Rust", "LangGraph");
    }

    @Test
    void requestsJsonModeFromTheProvider() throws Exception {
        llmAnswers("{\"nodes\":[],\"edges\":[]}", "stop");

        service.callLlmForExtraction(articles());

        ArgumentCaptor<HttpRequest> request = ArgumentCaptor.forClass(HttpRequest.class);
        verify(httpClient).send(request.capture(), any());
        JsonNode body = MAPPER.readTree(bodyOf(request.getValue()));
        assertThat(body.path("response_format").path("type").asText()).isEqualTo("json_object");
        assertThat(body.path("max_tokens").asInt()).isEqualTo(4000);
    }

    /** Drains the request's BodyPublisher; HttpRequest exposes no direct accessor for it. */
    private static String bodyOf(HttpRequest request) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        CompletableFuture<Void> done = new CompletableFuture<>();
        request.bodyPublisher().orElseThrow().subscribe(new Flow.Subscriber<ByteBuffer>() {
            @Override public void onSubscribe(Flow.Subscription s) { s.request(Long.MAX_VALUE); }
            @Override public void onNext(ByteBuffer item) {
                byte[] bytes = new byte[item.remaining()];
                item.get(bytes);
                out.writeBytes(bytes);
            }
            @Override public void onError(Throwable t) { done.completeExceptionally(t); }
            @Override public void onComplete() { done.complete(null); }
        });
        done.get();
        return out.toString(java.nio.charset.StandardCharsets.UTF_8);
    }
}
