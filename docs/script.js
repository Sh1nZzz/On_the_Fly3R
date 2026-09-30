const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const videos = document.querySelectorAll("video");
if (reduceMotion) {
  document.querySelectorAll(".video-toggle").forEach((button) => {
    button.textContent = "Play";
    button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Pause", "Play"));
  });
}

const loadVideo = (video) => {
  if (video.dataset.loaded) return;
  video.querySelectorAll("source[data-src]").forEach((source) => {
    source.src = source.dataset.src;
  });
  video.load();
  video.dataset.loaded = "true";
};

if ("IntersectionObserver" in window) {
  const videoObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          loadVideo(video);
          if (!reduceMotion) video.play().catch(() => {});
        } else {
          video.pause();
        }
      });
    },
    { rootMargin: "240px 0px", threshold: 0.1 },
  );
  videos.forEach((video) => videoObserver.observe(video));
} else {
  videos.forEach((video) => loadVideo(video));
}

document.querySelectorAll(".video-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    const video = button.parentElement.querySelector("video");
    loadVideo(video);
    if (video.paused) {
      video.play().catch(() => {});
      button.textContent = "Pause";
      button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Play", "Pause"));
    } else {
      video.pause();
      button.textContent = "Play";
      button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Pause", "Play"));
    }
  });
});

const copyButton = document.querySelector(".copy-button");
copyButton.addEventListener("click", async () => {
  const citation = document.querySelector(".citation-box code").textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = "Copied";
    setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 1800);
  } catch {
    copyButton.textContent = "Select to copy";
  }
});
