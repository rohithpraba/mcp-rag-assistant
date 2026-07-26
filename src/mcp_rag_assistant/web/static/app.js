const sampleQuestions = [
  "What technologies does this project use?",
  "How does the source refresh process work?",
  "What did the RAG benchmark measure?",
  "How did LoRA change the held-out behaviour results?",
  "Why was MCP stdio selected?",
  "What are the project's known limitations?",
];

const question = document.querySelector("#question");
const askButton = document.querySelector("#ask");
const status = document.querySelector("#status");
const answer = document.querySelector("#answer");
const answerState = document.querySelector("#answer-state");
const evidence = document.querySelector("#evidence");
const evidenceCount = document.querySelector("#evidence-count");

function setRepositoryLink(url) {
  for (const link of [document.querySelector("#github"), document.querySelector("#footer-github")]) {
    link.href = url;
    link.hidden = false;
  }
}

function setReadiness(isReady) {
  const indicator = document.querySelector("#ready");
  indicator.textContent = isReady ? "System ready" : "System unavailable";
  indicator.className = `readiness ${isReady ? "ready" : "not-ready"}`;
}

for (const text of sampleQuestions) {
  const button = document.createElement("button");
  button.className = "sample";
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", () => {
    question.value = text;
    question.focus();
  });
  document.querySelector("#samples").append(button);
}

fetch("/readyz")
  .then((response) => response.json())
  .then((payload) => setReadiness(Boolean(payload.ready)))
  .catch(() => setReadiness(false));

fetch("/api/v1/demo/sources")
  .then((response) => {
    if (!response.ok) throw new Error("Sources unavailable");
    return response.json();
  })
  .then((payload) => {
    if (payload.github_repository_url) setRepositoryLink(payload.github_repository_url);
  })
  .catch(() => {});

function renderEvidence(sources) {
  evidence.replaceChildren();
  evidenceCount.textContent = `${sources.length} ${sources.length === 1 ? "source" : "sources"}`;

  for (const source of sources) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const meta = document.createElement("div");
    const citation = document.createElement("span");
    const similarity = document.createElement("span");
    const excerpt = document.createElement("p");

    summary.textContent = `${source.label} · ${source.citation}`;
    meta.className = "source-meta";
    citation.textContent = `Citation ${source.citation}`;
    const score = Number(source.similarity);
    similarity.textContent = Number.isFinite(score) ? `Similarity ${score.toFixed(3)}` : "Similarity unavailable";
    excerpt.className = "source-excerpt";
    excerpt.textContent = source.text;
    meta.append(citation, similarity);
    details.append(summary, meta, excerpt);
    evidence.append(details);
  }
}

async function askQuestion() {
  const value = question.value.trim();
  if (!value) {
    status.className = "status error";
    status.textContent = "Enter a question before asking.";
    question.focus();
    return;
  }

  askButton.disabled = true;
  status.className = "status";
  status.textContent = "Retrieving evidence and generating an answer…";
  answer.className = "answer";
  answer.replaceChildren();
  answerState.className = "answer-state";
  answerState.textContent = "Working";
  evidence.replaceChildren();
  evidenceCount.textContent = "0 sources";

  try {
    const response = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value, top_k: 3 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The request could not be completed.");

    answer.textContent = payload.answer;
    if (payload.insufficient_evidence) {
      answer.classList.add("insufficient");
      answerState.className = "answer-state insufficient";
      answerState.textContent = "Insufficient evidence";
    } else {
      answerState.className = "answer-state valid";
      answerState.textContent = "Citations validated";
    }
    renderEvidence(payload.sources || []);
    status.textContent = "";
  } catch (error) {
    answer.innerHTML = '<p class="empty-state">No answer was returned.</p>';
    answerState.textContent = "Request failed";
    status.className = "status error";
    status.textContent = error instanceof Error ? error.message : "The request could not be completed.";
  } finally {
    askButton.disabled = false;
  }
}

askButton.addEventListener("click", askQuestion);
question.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") askQuestion();
});
