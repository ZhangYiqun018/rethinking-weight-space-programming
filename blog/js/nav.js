// Navigation for blog_response/index.html:
// (1) Floating table of contents that never covers content. Its mode comes from measuring where the page's content
//     is, on load, resize, font load, language switch and any change of the article's size, never from a breakpoint:
//     - sidebar: a fixed panel in the left margin, if the margin beside the article is wide enough for it;
//     - button: a fixed button in the left margin (label, icon or edge tab, whichever fits) that opens a drawer;
//       it moves to the other corner, or hides, while the spot it would take has content under it;
//     - inline: a collapsible box after the TL;DR, if even the edge tab does not fit;
// (2) Active section highlighting with IntersectionObserver + scroll position tracking, aria-current, and URL hash sync;
// (3) Smooth scrolling for TOC and cross-links;
// (4) Floating back-to-top button (appears after scrolling one screen) with bilingual tooltip/aria-label.
(() => {
  // Sidebar: from EDGE px off the window's edge to GAP px before the content; narrower than MIN px it is not used.
  const SIDEBAR = { min: 200, max: 300, edge: 16, gap: 28, top: 24, bottom: 24 };
  // Button: EDGE px off the window's edge, at least GAP px clear of content; the tab sits on the edge itself.
  const BUTTON = { edge: 12, gap: 12, tabGap: 4, bottom: 22, top: 16 };
  // Elements that can be wider than the block holding them.
  const WIDE_SELECTOR = "figure, table, img, svg, video, canvas, iframe, pre, .l-page, .l-page-outset, .l-screen";

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function currentLang() {
    return document.documentElement.getAttribute("data-lang") === "zh" ? "zh" : "en";
  }

  function init() {
    const tocNav = document.getElementById("toc");
    const toggle = document.getElementById("toc-toggle");
    const panel = document.getElementById("toc-panel");
    const backdrop = document.getElementById("toc-backdrop");
    const closeBtn = document.getElementById("toc-close");
    const backToTopBtn = document.getElementById("back-to-top");
    const article = document.querySelector("d-article");

    if (!tocNav || !toggle || !panel || !backdrop || !closeBtn || !article) return;

    // The nav's place in the article, for inline mode.
    const home = document.createComment(" table of contents ");
    tocNav.parentNode.insertBefore(home, tocNav);

    // 1. Bilingual aria-labels / tooltips synced with html[data-lang]
    function syncLangLabels() {
      const zh = currentLang() === "zh";
      tocNav.setAttribute("aria-label", zh ? "目录" : "Table of contents");
      toggle.setAttribute("title", zh ? "目录" : "Contents");
      panel.setAttribute("aria-label", zh ? "目录" : "Contents");
      const close = zh ? "关闭目录" : "Close contents";
      closeBtn.setAttribute("aria-label", close);
      closeBtn.setAttribute("title", close);
      if (backToTopBtn) {
        const label = zh ? "返回顶部" : "Back to top";
        backToTopBtn.setAttribute("aria-label", label);
        backToTopBtn.setAttribute("title", label);
      }
    }
    syncLangLabels();

    // 2. Where the content is. Boxes are in page coordinates vertically and window coordinates horizontally.
    // A block as wide as the window is a wrapper (a grid row, Distill's appendix lists): its children are measured
    // instead, and the block itself only if it paints a background (the title band).
    let layout = null;

    function paintsBackground(cs) {
      if (cs.backgroundImage && cs.backgroundImage !== "none") return true;
      const m = cs.backgroundColor.match(/rgba?\(([^)]+)\)/);
      if (!m) return false;
      const parts = m[1].split(",");
      return parts.length < 4 || parseFloat(parts[3]) > 0;
    }

    function measure() {
      const cw = document.documentElement.clientWidth;
      const sy = window.scrollY;
      const boxes = [];
      const add = (r, fullWidth) => {
        if (r.width > 0 && r.height > 0) {
          boxes.push({ left: r.left, right: r.right, top: r.top + sy, bottom: r.bottom + sy, fullWidth });
        }
      };
      function visit(el) {
        if (el === tocNav) return;
        const cs = getComputedStyle(el);
        if (cs.display === "none" || cs.position === "fixed") return;
        const r = el.getBoundingClientRect();
        if (cs.display === "contents" || r.width >= cw - 1) {
          if (cs.display !== "contents" && cs.visibility !== "hidden" && paintsBackground(cs)) add(r, true);
          Array.from(el.children).forEach(visit);
          return;
        }
        if (cs.visibility !== "hidden") add(r);
        el.querySelectorAll(WIDE_SELECTOR).forEach((child) => {
          if (getComputedStyle(child).visibility !== "hidden") add(child.getBoundingClientRect());
        });
      }
      Array.from(document.body.children).forEach(visit);

      const tldr = document.querySelector(".tldr") || article;
      const ar = article.getBoundingClientRect();
      const bandTop = tldr.getBoundingClientRect().top + sy;
      const bandBottom = ar.bottom + sy;
      let bandLeft = cw;
      let pageLeft = cw;
      // A full-width band (the title) leaves no room beside it for the button to fit; the button just avoids it
      // (see placeButton). The sidebar, which cannot move out of the way, must not have one beside it at all.
      boxes.forEach((b) => {
        if (b.bottom > bandTop && b.top < bandBottom) bandLeft = Math.min(bandLeft, b.left);
        if (!b.fullWidth) pageLeft = Math.min(pageLeft, b.left);
      });
      return { boxes, bandTop, bandBottom, bandLeft, pageLeft, cw };
    }

    // 3. Mode: sidebar if it fits beside the article (the only part of the page it can reach), else the largest
    // button form that fits beside the whole page, else beside the article, else inline.
    let mode = null;
    let form = null;
    let sidebarBox = null;
    let buttonBoxes = [];
    let buttonSize = { w: 0, h: 0 };
    let buttonLeft = 0;
    let open = false;
    let inlineOpen = false;

    function moveNav(toInline) {
      if (toInline) {
        if (tocNav.parentNode !== home.parentNode || tocNav.previousSibling !== home) {
          home.parentNode.insertBefore(tocNav, home.nextSibling);
        }
      } else if (tocNav.parentNode !== document.body) {
        document.body.insertBefore(tocNav, article);
      }
    }

    function measureButtonForm(f) {
      toggle.dataset.form = f;
      return { w: toggle.offsetWidth, h: toggle.offsetHeight };
    }

    function chooseButtonForm(room) {
      const forms = [];
      ["pill", "icon", "tab"].forEach((f) => {
        const size = measureButtonForm(f);
        const left = f === "tab" ? 0 : BUTTON.edge;
        const gap = f === "tab" ? BUTTON.tabGap : BUTTON.gap;
        forms.push({ f, size, left, need: left + size.w + gap });
      });
      for (const x of [room.pageLeft, room.bandLeft]) {
        const fit = forms.find((c) => c.need <= x);
        if (fit) return fit;
      }
      return null;
    }

    function setMode(next) {
      if (mode === next) return;
      if (mode === "button" && open) setOpen(false, false);
      mode = next;
      tocNav.classList.toggle("is-sidebar", mode === "sidebar");
      tocNav.classList.toggle("is-button", mode === "button");
      tocNav.classList.toggle("is-inline", mode === "inline");
      if (mode !== "sidebar") {
        panel.style.left = panel.style.top = panel.style.width = panel.style.maxHeight = "";
      }
      if (mode !== "button") {
        toggle.style.left = toggle.style.top = "";
        toggle.classList.remove("is-obstructed");
      }
      if (mode === "inline") toggle.removeAttribute("data-form");
    }

    function applyLayout() {
      layout = measure();
      const sidebarWidth = Math.min(SIDEBAR.max, layout.bandLeft - SIDEBAR.gap - SIDEBAR.edge);
      if (sidebarWidth >= SIDEBAR.min) {
        moveNav(false);
        setMode("sidebar");
        toggle.hidden = true;
        backdrop.hidden = true;
        closeBtn.hidden = true;
        panel.hidden = false;
        panel.removeAttribute("role");
        panel.removeAttribute("aria-modal");
        toggle.removeAttribute("data-form");
        sidebarBox = { left: layout.bandLeft - SIDEBAR.gap - sidebarWidth, width: sidebarWidth };
        panel.style.left = `${sidebarBox.left}px`;
        panel.style.width = `${sidebarBox.width}px`;
        placeSidebar(true);
        return;
      }
      // Size the button out of sight, so measuring never shows a half-placed state.
      moveNav(false);
      toggle.hidden = false;
      toggle.style.visibility = "hidden";
      setMode("button");
      const fit = chooseButtonForm(layout);
      if (fit) {
        form = fit.f;
        toggle.dataset.form = form;
        buttonSize = fit.size;
        buttonLeft = fit.left;
        toggle.style.left = `${buttonLeft}px`;
        toggle.style.visibility = "";
        // Only content that reaches the button's column can ever be under it.
        const reach = buttonLeft + buttonSize.w + BUTTON.gap;
        buttonBoxes = layout.boxes.filter((b) => b.left < reach && b.right > buttonLeft - BUTTON.gap);
        closeBtn.hidden = false;
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-modal", "true");
        if (!open) {
          panel.hidden = true;
          backdrop.hidden = true;
        }
        placeButton();
        return;
      }
      toggle.style.visibility = "";
      moveNav(true);
      setMode("inline");
      closeBtn.hidden = true;
      backdrop.hidden = true;
      panel.removeAttribute("role");
      panel.removeAttribute("aria-modal");
      panel.hidden = !inlineOpen;
      toggle.setAttribute("aria-expanded", inlineOpen ? "true" : "false");
    }

    // Sidebar: top aligned with the TL;DR, then held SIDEBAR.top px from the window's top; it never goes past the
    // end of the article, and scrolls on its own when taller than the window.
    let lastMaxHeight = -1;
    function placeSidebar(force) {
      if (mode !== "sidebar" || !layout) return;
      const sy = window.scrollY;
      let top = Math.max(SIDEBAR.top, layout.bandTop - sy);
      const maxHeight = Math.max(120, Math.floor(window.innerHeight - top - SIDEBAR.bottom));
      if (force || maxHeight !== lastMaxHeight) {
        panel.style.maxHeight = `${maxHeight}px`;
        lastMaxHeight = maxHeight;
      }
      const height = Math.min(panel.scrollHeight, maxHeight);
      const end = layout.bandBottom - sy;
      if (top + height > end) top = end - height;
      panel.style.top = `${Math.round(top)}px`;
    }

    // Button: bottom-left, else top-left, whichever has no content under it (with BUTTON.gap of air); else hidden.
    function placeButton() {
      if (mode !== "button" || !layout) return;
      const sy = window.scrollY;
      const gap = form === "tab" ? BUTTON.tabGap : BUTTON.gap;
      const spots = [window.innerHeight - BUTTON.bottom - buttonSize.h, BUTTON.top];
      const free = spots.find((top) => {
        const t = top + sy - gap;
        const b = top + sy + buttonSize.h + gap;
        const l = buttonLeft - gap;
        const r = buttonLeft + buttonSize.w + gap;
        return !buttonBoxes.some((x) => x.left < r && x.right > l && x.top < b && x.bottom > t);
      });
      if (free === undefined && !open) {
        toggle.classList.add("is-obstructed");
        if (document.activeElement === toggle) toggle.blur();
        return;
      }
      toggle.classList.remove("is-obstructed");
      if (free !== undefined) toggle.style.top = `${Math.round(free)}px`;
    }

    // 4. Drawer (button mode) and inline box
    function focusables() {
      return Array.from(panel.querySelectorAll("a[href], button")).filter((el) => el.getClientRects().length > 0);
    }

    function setOpen(next, restoreFocus) {
      if (mode === "inline") {
        inlineOpen = next;
        panel.hidden = !next;
        toggle.setAttribute("aria-expanded", next ? "true" : "false");
        return;
      }
      if (mode !== "button" || open === next) return;
      open = next;
      tocNav.classList.toggle("is-open", next);
      panel.hidden = !next;
      backdrop.hidden = !next;
      toggle.setAttribute("aria-expanded", next ? "true" : "false");
      if (next) {
        revealActiveLink(true);
        const target = (activeId && linkById.get(activeId)) || closeBtn;
        try {
          target.focus({ preventScroll: true });
        } catch (e) {}
      } else {
        placeButton();
        if (restoreFocus && !toggle.classList.contains("is-obstructed")) {
          try {
            toggle.focus({ preventScroll: true });
          } catch (e) {}
        }
      }
    }

    toggle.addEventListener("click", () => {
      setOpen(mode === "inline" ? !inlineOpen : !open, true);
    });
    closeBtn.addEventListener("click", () => setOpen(false, true));
    backdrop.addEventListener("click", () => setOpen(false, true));
    document.addEventListener("keydown", (e) => {
      if (!open) return;
      if (e.key === "Escape") {
        e.preventDefault();
        setOpen(false, true);
      } else if (e.key === "Tab") {
        // Keep focus inside the drawer while it is open.
        const items = focusables();
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        const inside = panel.contains(document.activeElement);
        if (e.shiftKey && (document.activeElement === first || !inside)) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && (document.activeElement === last || !inside)) {
          e.preventDefault();
          first.focus();
        }
      }
    });

    // 5. Collect TOC entries and target elements in document order
    const tocLinks = Array.from(tocNav.querySelectorAll('a[href^="#"]'));
    const entries = [];
    const linkById = new Map();

    tocLinks.forEach((link) => {
      const rawHash = link.getAttribute("href") || "";
      const id = rawHash.slice(1);
      if (!id) return;
      const target = document.getElementById(id);
      if (!target) return;
      entries.push({ id, link, target });
      linkById.set(id, link);
    });

    let activeId = null;
    let hashTimer = null;
    let programmaticScrollUntil = 0;

    // Keep the active entry visible inside the sidebar or drawer when they scroll on their own.
    function revealActiveLink(center) {
      const link = activeId && linkById.get(activeId);
      if (!link || panel.hidden || panel.scrollHeight <= panel.clientHeight + 1) return;
      const pr = panel.getBoundingClientRect();
      const lr = link.getBoundingClientRect();
      const margin = 32;
      if (center || lr.top < pr.top + margin || lr.bottom > pr.bottom - margin) {
        panel.scrollTop += lr.top - pr.top - (panel.clientHeight - lr.height) / 2;
      }
    }

    function updateUrlHash(id, push) {
      if (!window.history || !window.history.replaceState) return;
      const base = window.location.pathname + window.location.search;
      const nextUrl = id ? `${base}#${id}` : base;
      if (window.location.hash === (id ? `#${id}` : "")) return;
      try {
        if (push && window.history.pushState) {
          window.history.pushState(null, "", nextUrl);
        } else {
          window.history.replaceState(null, "", nextUrl);
        }
      } catch (e) {}
    }

    function setActive(id, syncHash) {
      if (!id || id === activeId) return;
      activeId = id;

      tocLinks.forEach((a) => {
        a.classList.remove("is-active");
        a.removeAttribute("aria-current");
      });
      tocNav.querySelectorAll("li.is-active-branch").forEach((li) => {
        li.classList.remove("is-active-branch");
      });

      const activeLink = linkById.get(id);
      if (activeLink) {
        activeLink.classList.add("is-active");
        activeLink.setAttribute("aria-current", "location");
        let p = activeLink.parentElement;
        while (p && p !== tocNav) {
          if (p.tagName === "LI") p.classList.add("is-active-branch");
          p = p.parentElement;
        }
        if (mode === "sidebar") revealActiveLink(false);
      }

      if (syncHash && Date.now() >= programmaticScrollUntil) {
        clearTimeout(hashTimer);
        hashTimer = setTimeout(() => {
          if (window.scrollY < 160) {
            updateUrlHash("", false);
          } else {
            updateUrlHash(activeId, false);
          }
        }, 140);
      }
    }

    function computeActiveFromScroll() {
      if (!entries.length) return;
      if (Date.now() < programmaticScrollUntil) return;

      // A heading becomes the active section once it has passed 30% of the window's height.
      const triggerY = window.innerHeight * 0.3;
      const atBottom =
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 24;

      if (atBottom) {
        for (let i = entries.length - 1; i >= 0; i--) {
          const r = entries[i].target.getBoundingClientRect();
          if (r.top < window.innerHeight * 0.8) {
            setActive(entries[i].id, true);
            return;
          }
        }
      }

      let chosen = entries[0].id;
      for (let i = 0; i < entries.length; i++) {
        const top = entries[i].target.getBoundingClientRect().top;
        if (top <= triggerY) {
          chosen = entries[i].id;
        } else {
          break;
        }
      }
      setActive(chosen, window.scrollY >= 160);
    }

    // Observe all section headings with IntersectionObserver
    if ("IntersectionObserver" in window && entries.length) {
      const io = new IntersectionObserver(
        () => {
          computeActiveFromScroll();
        },
        {
          root: null,
          rootMargin: "0px 0px -70% 0px",
          threshold: [0, 1]
        }
      );
      entries.forEach((entry) => io.observe(entry.target));
    }

    // 6. Smooth scrolling for all in-page anchor links (#toc, #table-claims-summary, .tldr, .back-to-summary)
    document.addEventListener("click", (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const anchor = e.target.closest ? e.target.closest('a[href^="#"]') : null;
      if (!anchor) return;
      const href = anchor.getAttribute("href");
      if (!href || href === "#") return;
      const targetId = href.slice(1);
      const targetEl = document.getElementById(targetId);
      if (!targetEl) return;

      e.preventDefault();

      // Choosing an entry closes the drawer, or the inline box.
      if (tocNav.contains(anchor)) {
        if (open) setOpen(false, false);
        if (mode === "inline") setOpen(false, false);
      }

      programmaticScrollUntil = Date.now() + 700;
      if (linkById.has(targetId)) {
        setActive(targetId, false);
      }
      updateUrlHash(targetId, true);

      targetEl.scrollIntoView({
        behavior: prefersReducedMotion() ? "auto" : "smooth",
        block: "start"
      });

      // Move keyboard focus to target heading without jumping scroll position
      if (!targetEl.hasAttribute("tabindex")) {
        targetEl.setAttribute("tabindex", "-1");
      }
      try {
        targetEl.focus({ preventScroll: true });
      } catch (err) {}
    });

    // 7. Back-to-top button visibility and click action
    function syncBackToTop() {
      if (!backToTopBtn) return;
      const show = window.scrollY > window.innerHeight;
      if (show) {
        backToTopBtn.hidden = false;
        backToTopBtn.classList.add("is-visible");
      } else {
        backToTopBtn.classList.remove("is-visible");
        backToTopBtn.hidden = true;
      }
    }

    if (backToTopBtn) {
      backToTopBtn.addEventListener("click", () => {
        programmaticScrollUntil = Date.now() + 700;
        if (entries.length) {
          setActive(entries[0].id, false);
        }
        updateUrlHash("", true);
        window.scrollTo({
          top: 0,
          behavior: prefersReducedMotion() ? "auto" : "smooth"
        });
      });
    }

    // 8. Re-measure on anything that can move content: resize, fonts, language, images and figures loading.
    let layoutPending = false;
    function scheduleLayout() {
      if (layoutPending) return;
      layoutPending = true;
      window.requestAnimationFrame(() => {
        layoutPending = false;
        applyLayout();
      });
    }

    let rafPending = false;
    function onScroll() {
      if (rafPending) return;
      rafPending = true;
      window.requestAnimationFrame(() => {
        rafPending = false;
        placeSidebar(false);
        placeButton();
        computeActiveFromScroll();
        syncBackToTop();
      });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", () => {
      scheduleLayout();
      onScroll();
    });
    window.addEventListener("load", scheduleLayout);
    if (document.fonts) {
      if (document.fonts.ready) document.fonts.ready.then(scheduleLayout);
      if (document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", scheduleLayout);
    }
    new MutationObserver(() => {
      syncLangLabels();
      scheduleLayout();
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-lang", "lang"]
    });
    // The page's height changes whenever content does; Distill adds <d-appendix> (references) after this runs.
    if ("ResizeObserver" in window) {
      const ro = new ResizeObserver(scheduleLayout);
      ro.observe(document.documentElement);
      ro.observe(article);
    }
    new MutationObserver(scheduleLayout).observe(document.body, { childList: true });

    // Initial state on load
    applyLayout();
    if (window.location.hash && linkById.has(window.location.hash.slice(1))) {
      setActive(window.location.hash.slice(1), false);
    } else {
      computeActiveFromScroll();
    }
    revealActiveLink(false);
    syncBackToTop();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
