import { expect, test } from "@playwright/test";

const graphFixture = {
  nodes: [
    {
      id: "1",
      label: "LangGraph",
      category: "Framework",
      hypeScore: 4.5,
      mentionCount: 12,
      summary: "LangGraph is a framework for building stateful LLM workflows.",
      sourceTitle: "LangGraph in production",
      sourceUrl: "https://news.ycombinator.com/item?id=1",
      sourcePlatform: "hackernews",
    },
  ],
  edges: [],
  meta: { days: 7, nodeCount: 1, edgeCount: 0, generatedAt: new Date().toISOString() },
};

const articlesFixture = { articles: [], meta: { days: 7, count: 0, generatedAt: new Date().toISOString() } };

test.beforeEach(async ({ page }) => {
  await page.route("**/api/v1/graph*", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(graphFixture) }),
  );
  await page.route("**/api/v1/articles*", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(articlesFixture) }),
  );
});

test("skip-link is reachable via Tab and jumps to main", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  const skipLink = page.getByRole("link", { name: /skip to content/i });
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
});

test("slash focuses the search input", async ({ page }) => {
  await page.goto("/");
  // GraphView is client-only, so its keydown listener attaches after "load".
  // Wait for the graph to render, or a slow runner presses "/" too early.
  await expect(page.getByText("LangGraph").first()).toBeVisible();
  await page.keyboard.press("/");
  const search = page.getByRole("textbox", { name: /search technology/i });
  await expect(search).toBeFocused();
});

test("Escape closes the modal and restores focus", async ({ page }) => {
  await page.goto("/");
  await page.getByText("LangGraph").first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
});

test("view mode toggle: g c keeps the tab in columns", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Grid" }).click();
  await expect(page.getByRole("tab", { name: "Grid" })).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("g");
  await page.keyboard.press("c");
  await expect(page.getByRole("tab", { name: "Columns" })).toHaveAttribute("aria-selected", "true");
});
