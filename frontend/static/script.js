<script>
  document.addEventListener("DOMContentLoaded", async () => {
    const loading = document.getElementById("loading");
    const content = document.getElementById("resultsContent");

    // Show loading spinner
    loading.style.display = "flex";

    const stored = localStorage.getItem("reviewInput");
    if (!stored) {
      alert("No input found. Please go back to the home page.");
      window.location.href = "index.html";
      return;
    }

    const { title, inclusion, exclusion } = JSON.parse(stored);

    try {
      const response = await fetch("http://127.0.0.1:8000/generate-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          inclusion_criteria: inclusion.split(',').map(x => x.trim()),
          exclusion_criteria: exclusion.split(',').map(x => x.trim())
        })
      });

      const data = await response.json();

      // Fill in summary
      document.getElementById("summaryText").innerText = data.review_summary;

      // Fill in papers
      const paperContainer = document.getElementById("papersList");
      data.papers.forEach((paper, index) => {
        const paperDiv = document.createElement("div");
        paperDiv.className = "paper";
        paperDiv.innerHTML = `
          <h3>${index + 1}. ${paper.title}</h3>
          <p>${paper.abstract}</p>
          <p class="score">Relevance Score: ${paper.score}</p>
          <a href="${paper.url}" target="_blank">View Paper</a>
        `;
        paperContainer.appendChild(paperDiv);
      });

      // Hide loader, show content
      loading.style.display = "none";
      content.style.display = "flex";
    } catch (err) {
      alert("Error loading results. Please try again.");
      console.error(err);
    }
  });
</script>
