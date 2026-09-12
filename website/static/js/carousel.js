(() => {
  const dialog = document.querySelector('.video-dialog');
  const player = dialog.querySelector('video');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const carousels = [];
  let opener;
  const sync = () => carousels.forEach(c => c.sync());
  document.querySelectorAll('.video-carousel').forEach(root => {
    const track = root.querySelector('.carousel-track');
    const cards = [...root.querySelectorAll('.carousel-card')];
    const videos = cards.map(card => card.querySelector('video'));
    const previous = root.querySelector('.previous');
    const next = root.querySelector('.next');
    let paused = reduced.matches;
    const visible = new Set();
    const playback = () => videos.forEach(video => {
      if (visible.has(video) && !paused && !dialog.open && !document.hidden) video.play().catch(() => {});
      else video.pause();
    });
    carousels.push({sync: playback});
    const observer = new IntersectionObserver(entries => {
      // Observe the card button, not the video: zoomed videos are clipped by the card and would never reach the threshold.
      entries.forEach(entry => { const video = entry.target.querySelector('video'); entry.isIntersecting && entry.intersectionRatio >= .5 ? visible.add(video) : visible.delete(video); });
      playback();
    }, {threshold: [0, .5]});
    videos.forEach(v => { v.muted = true; observer.observe(v.closest('.video-open')); });
    const position = () => cards.map(card => card.offsetLeft - cards[0].offsetLeft);
    const go = i => track.scrollTo({left: position()[Math.max(0, Math.min(i, cards.length - 1))], behavior: reduced.matches ? 'instant' : 'smooth'});
    const active = () => position().reduce((best, value, i, list) => Math.abs(value-track.scrollLeft) < Math.abs(list[best]-track.scrollLeft) ? i : best, 0);
    const update = () => {
      previous.disabled = track.scrollLeft < 3;
      next.disabled = track.scrollLeft >= track.scrollWidth - track.clientWidth - 3;

    };
    previous.onclick = () => go(active() - 1);
    next.onclick = () => go(active() + 1);
    track.addEventListener('keydown', event => {
      if (event.target !== track) return;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); go(active() + (event.key === 'ArrowRight' ? 1 : -1)); }
    });
    track.addEventListener('scroll', update, {passive: true});
    new ResizeObserver(update).observe(track);
    reduced.addEventListener('change', () => { paused = reduced.matches; playback(); });
    cards.forEach(card => card.querySelector('.video-open').addEventListener('click', event => {
      opener = event.currentTarget;
      dialog.setAttribute('aria-label', opener.dataset.title);
      player.src = card.querySelector('source').src;
      player.muted = false;
      dialog.showModal(); document.body.classList.add('dialog-open'); sync();
      player.play().catch(() => {});
    }));
    update();
  });
  dialog.querySelector('.dialog-close').onclick = () => dialog.close();
  dialog.addEventListener('click', event => { if (event.target === dialog) {const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) dialog.close();} });
  dialog.addEventListener('close', () => {player.pause();player.removeAttribute('src');player.load();document.body.classList.remove('dialog-open');opener?.focus({preventScroll:true});sync();});
  document.addEventListener('visibilitychange', sync);
})();
