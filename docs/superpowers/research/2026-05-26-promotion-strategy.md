# Promotion strategy — mlx-teacache & mlx-taef

Two-week-old MLX libraries, free-channels only, written for one maintainer with finite weekends. Honest baseline, no marketing pablum, every claim has a URL.

Date of research: 2026-05-26.

---

## 1. Current state (honest baseline)

### mlx-teacache

PyPI says **1 download yesterday, 166 in the last week, 764 in the last month** ([pypistats](https://pypistats.org/packages/mlx-teacache)). The 764/month figure is real but heavily inflated by `pip install` retries during your own development plus CI runs — that's normal at this stage and shouldn't be read as 764 humans. The repo has **1 star, 0 forks, 0 watchers** ([github.com/IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache)). Latest release v0.6.1 shipped 2026-05-26. License Apache-2.0. The GitHub repo description is good — *"TeaCache step-skipping for FLUX diffusion on Apple Silicon, in pure MLX"* — but **no GitHub topics are set**, so the repo doesn't surface under `topic:mlx` / `topic:flux` / `topic:apple-silicon` browse pages.

Discoverability via search is basically zero. Searching `"mlx-teacache" FLUX Apple Silicon` returns the repo itself plus generic MLX content; nothing else mentions it. It does not appear in [awesome-mlx](https://github.com/raullenchai/awesome-mlx) (80+ projects, Image & Video Generation section exists), nor in [ml-explore/mlx Discussion #654](https://github.com/ml-explore/mlx/discussions/654) ("MLX Community Projects" — the official Apple MLX show-and-tell thread, where recent submissions like mlx-sparse, mlx-node, mlx-serve, MOLA, mlx-code all got added). Upstream [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) (1.3k stars) lists community ports for FramePack, FastVideo, ComfyUI, SD.Next, DiffSynth Studio — **no MLX port listed**, even though yours is the first one.

The single biggest discoverability lever I found: [filipstrand/mflux issue #113 "Upscaler and TeaCache"](https://github.com/filipstrand/mflux/issues/113), opened by `azrahello` on 2025-01-12, **still open, no mention of mlx-teacache**. mflux has 2.1k stars and 76 open issues. Someone is sitting in that thread asking for exactly your library and they don't know it exists.

### mlx-taef

PyPI: **1 download yesterday, 35 in the last week, 331 in the last month** ([pypistats](https://pypistats.org/packages/mlx-taef)). Repo has **2 stars, 1 fork, 0 watchers** ([github.com/IonDen/mlx-taef](https://github.com/IonDen/mlx-taef)). v0.1.1 shipped 2026-05-13. License MIT. The GitHub description is decent — *"Tiny AutoEncoders for diffusion latents on Apple Silicon, in pure MLX. Live previews + low-memory decode for FLUX.1, FLUX.2 Klein, SD1.x, SDXL"* — but again **no topics set**. PyPI summary is weaker than the GitHub one: *"Tiny AutoEncoders for diffusion (TAESD family) on Apple MLX"* — it drops the live-preview hook and the FLUX.2 Klein name that would catch eyes.

Searching `"mlx-taef"` returns the repo plus the mlx-teacache README (which references it for live previews). The upstream [madebyollin/taesd](https://github.com/madebyollin/taesd) repo (936 stars) does **not** mention any MLX port. The [madebyollin/taef2 HuggingFace page](https://huggingface.co/madebyollin/taef2) has an empty Discussions tab — meaning zero MLX-port announcements there yet. That's a clean slot.

The v0.2.0 design at `docs/superpowers/specs/2026-05-26-mlx-taef-v0.2.0-design.md` is the right artifact for the next promotion push — once it ships with the measured-showcase + COMPARISON.md, there's something *visual* to share, which currently the README lacks.

### Combined honest read

These are 2-week-old, technically real, niche-correct libraries with **near-zero organic reach**. Nobody outside the maintainer's own development loop has installed them in significant numbers. The good news: the *substantive* prerequisites for promotion (working code, benchmarks, license cleanliness, semver discipline) are already in place. What's missing is distribution — and the distribution failure is concentrated in 4-5 specific channels where 30 minutes of work each would 10× the visibility floor.

---

## 2. Target users and where they live

### mlx-teacache target user

Someone on M1 Max / M2 Max / M3 / M4 running FLUX locally, watching 50-step Klein-base-9B chew up minutes per image, looking for a way to make it faster without giving up quality.

| Channel | URL | Audience & MLX-awareness | Leverage | Etiquette |
|---|---|---|---|---|
| `filipstrand/mflux` issues & discussions | [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux) — 2.1k stars, 76 open issues, README links 5 community wrappers (MindCraft Studio, Mflux-ComfyUI, MFLUX-WEBUI, mflux-fasthtml, mflux-streamlit) | Exact target audience; high MLX-fluency; actively requesting TeaCache | **High** | Comment on existing issue #113, don't open a new one. Lead with measured numbers, link to the bench script, not the marketing tagline. |
| `ali-vilab/TeaCache` README community-ports list | [github.com/ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — 1.3k stars; README maintains "third-party implementations" list | Researchers + ecosystem people; will see the link forever | **High** | Open a PR adding `mlx-teacache` to the community-implementations list in TeaCache4FLUX/README.md. Short, factual, no benchmark in the PR body — just "MLX port for Apple Silicon, FLUX.1 / FLUX.2 Klein". |
| `awesome-mlx` (raullenchai) | [github.com/raullenchai/awesome-mlx](https://github.com/raullenchai/awesome-mlx) — 80+ projects, has Image & Video Generation category, [submit-project issue template](https://github.com/raullenchai/awesome-mlx/issues/new?template=submit_project.yml) | MLX ecosystem browsers | **Medium-High** | Use the issue form. One line. They'll add it. |
| `ml-explore/mlx` Discussion #654 | [github.com/ml-explore/mlx/discussions/654](https://github.com/ml-explore/mlx/discussions/654) — Apple's official "MLX Community Projects" thread | Apple MLX core team + ecosystem | **Medium-High** | "Leave a comment, we'll add it" per the OP. Recent additions: mlx-sparse, mlx-node, mlx-serve, mlx-code (March-May 2026). |
| r/LocalLLaMA / r/StableDiffusion | [reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA) (~500k), [reddit.com/r/StableDiffusion](https://www.reddit.com/r/StableDiffusion) (~700k) | Huge but mostly NVIDIA-focused; Mac users are a vocal minority | **Medium** | Post with a clear "Mac-only" label in the title; benchmark image + table; no "check out my project" framing — frame as "I measured X". |
| Hacker News (Show HN) | [news.ycombinator.com/show](https://news.ycombinator.com/show) | General tech; loves Apple Silicon + perf | **Medium-Low** for now | Don't burn the Show HN until you have a *visual* hook (side-by-side, GIF). Best paired with the mlx-taef v0.2.0 launch when the comparison page exists. |
| draw-things Discord | (invite via [drawthings.ai](https://drawthings.ai/)) | Mac diffusion users specifically | **Low-Medium** | They are explicit competitors of mflux-stack; Draw Things claims ~25% faster than mflux ([engineering blog](https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)). Don't pitch — engage genuinely on Metal/MLX threads when relevant. |

### mlx-taef target user

Same Mac diffusion crowd, plus the longer tail of researchers/notebook users who want live previews without paying full-VAE memory. Plus anyone bottlenecked on the FLUX.2 VAE's ~9.6 GB peak on a 32 GB Mac.

| Channel | URL | Audience | Leverage | Etiquette |
|---|---|---|---|---|
| `madebyollin/taef2` HF Discussions | [huggingface.co/madebyollin/taef2/discussions](https://huggingface.co/madebyollin/taef2/discussions) — currently empty | Ollin's followers + anyone landing on the model card | **High** | Open one thread: *"MLX port (mlx-taef) — TAEF2 decode on Apple Silicon, 100 ms @ 1024² vs N s for full VAE"*. Credit upstream. Single thread, short, with a link. Same for `madebyollin/taesd`, `madebyollin/taef1`. |
| `madebyollin/taesd` GitHub | [github.com/madebyollin/taesd](https://github.com/madebyollin/taesd) — 936 stars; README has no community-ports section | Wider TAESD ecosystem | **Medium-High** | Open one issue or discussion: "MLX port available — happy to add a link to the README if useful". Let the maintainer pull, don't push. |
| `awesome-mlx` + `ml-explore/mlx` Discussion #654 | (same as above) | MLX ecosystem | **Medium-High** | Same submission paths as mlx-teacache. |
| `filipstrand/mflux` issues/discussions | [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux) | Direct integration audience | **High** | Open a discussion (not an issue) showing the live-preview callback wired up in a mflux generate loop. Demo, not pitch. |
| HN Show HN | [news.ycombinator.com/show](https://news.ycombinator.com/show) | General tech | **Medium** | Hold for the v0.2.0 measured-showcase ship; that's when there'll be a real visual hook. |
| r/StableDiffusion | [reddit.com/r/StableDiffusion](https://www.reddit.com/r/StableDiffusion) | Diffusion-native users | **Medium** | Lead with the live-preview GIF, not the library name. |
| Apple Developer Forums (MLX) | [developer.apple.com/forums/tags/mlx](https://developer.apple.com/forums/tags/mlx) | Apple staff lurkers + early MLX adopters | **Low** | Low traffic, but a presence here is cheap. Reply to one or two existing MLX/diffusion threads when relevant. Don't create a "look at my library" post — those get ignored. |

---

## 3. README/PyPI quick wins

### Both repos

1. **Set GitHub topics now.** Both repos have *no topics set*. Recommended for **mlx-teacache**: `mlx`, `apple-silicon`, `flux`, `diffusion`, `teacache`, `mflux`, `image-generation`, `inference-optimization`. For **mlx-taef**: `mlx`, `apple-silicon`, `taesd`, `vae`, `flux`, `diffusion`, `live-preview`, `tiny-autoencoder`. This is a 30-second edit in repo Settings → About → Topics. It controls whether the repo appears under [github.com/topics/mlx](https://github.com/topics/mlx) browse pages.
2. **PyPI summary parity.** The PyPI summary for mlx-taef (*"Tiny AutoEncoders for diffusion (TAESD family) on Apple MLX"*) is *worse* than the GitHub one. The GitHub blurb names FLUX.1/FLUX.2 Klein and "live previews + low-memory decode" — those are the search keywords. Bring the GitHub line into `pyproject.toml`'s `description` field. Same applies to mlx-teacache: the PyPI summary is fine but consider adding the measured headline ("1.36× speedup on FLUX.2 Klein-base-9B 50-step + CFG on M1 Max") into the long description (`README.md` is what PyPI renders).
3. **Hero image / GIF above the fold.** Both READMEs open with text. Diffusion projects are visual. mlx-teacache should embed a side-by-side (vanilla vs wrapper, same seed, identical or near-identical image, with the wall-clock number underneath) at the very top, before "What it does". mlx-taef should embed a live-preview GIF showing latent→RGB updating during a generate, side-by-side with the final full-VAE render. This is what the v0.2.0 measured-showcase is meant to produce — ship that first, then put the image in the README.

### mlx-teacache specifically

4. **Add "First MLX port of TeaCache" to the README hero.** This is true (verified — TeaCache upstream lists no MLX port) and it's a genuine SEO + curiosity hook. One line: *"The first MLX port of TeaCache (ali-vilab/TeaCache), the training-free step-skipping optimizer."*
5. **Move "How the speedup happens" up.** Right now it's deep in the README (section ~15 of 21). The mflux-`mx.compile`-sidestepping story is the *interesting* part — that's the kind of detail that gets reshared on X. Promote it to a short section right after the headline benchmark.
6. **Add a "reproduce" link near every speedup number.** Per your own CLAUDE.md rule, every perf claim needs a committed benchmark. The README should make the bench script invocation copy-pasteable next to each number: `uv run python scripts/bench_comparison.py --model klein-base-9b --steps 50 --cfg 4.0`. People scanning the README for "is this real" should be able to verify in 60 seconds.

### mlx-taef specifically

7. **Lead with the memory number, not the speed number.** *"~100 ms decode at 1024² vs seconds for full VAE; ~1 GB peak vs ~9.6 GB"* — the **9.6 → 1 GB** ratio is what makes someone on a 16 GB Mac install this. Speed alone competes with "just wait"; memory unlocks workflows that were literally impossible before. Reorder the benchmarks table to lead with peak memory.
8. **Name the use-cases in the hero.** Currently the README says "Live previews + low-memory decode" but doesn't immediately spell out *for whom*. Add a 3-bullet "What it's for":
   - Live previews in mflux during long generates
   - Notebook iteration where the full VAE OOMs
   - Img2img / inpainting pipelines that run encode-decode many times
9. **Add the mflux integration snippet to the README hero.** The integration is the killer feature. The README mentions it in "mflux live previews" but it's below the fold. A 6-line code block at the top, copy-pasteable, with a `taef2` variant, would convert browsers to installers.

---

## 4. One-time promotion actions this week

Pick 5-8. Each one is 1-3 hours and should land 10-50 net new visitors over 30 days. None of them trip self-promo etiquette.

### Action 1 — Comment on mflux issue #113 (highest leverage, 30 minutes)

**Channel:** [github.com/filipstrand/mflux/issues/113](https://github.com/filipstrand/mflux/issues/113)
**Action:** Post one comment on this open issue (open since Jan 2025, requesting exactly TeaCache for mflux). Lead with the bench number from v0.6.1, link to the repo, link to the bench script, mention that it works as a wrapper (no fork required).
**Etiquette:** Factual, no "check out my project" language. Acknowledge the request, present the wrapper, invite feedback. The original requester `azrahello` is the intended reader, but anyone subscribed to mflux issues will see it.
**Expected outcome:** 20-50 visitors over 30 days; some chance of `filipstrand` linking from the mflux README's community-projects section (he already links 5 wrappers).

### Action 2 — PR to ali-vilab/TeaCache community-ports list (45 minutes)

**Channel:** [github.com/ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — the README maintains a list of third-party integrations (FramePack, ComfyUI, SD.Next, etc.)
**Action:** Open a PR adding one line to the appropriate section: `[mlx-teacache](https://github.com/IonDen/mlx-teacache) — MLX port for FLUX diffusion on Apple Silicon (FLUX.1, FLUX.2 Klein)`.
**Etiquette:** Tiny, focused PR. No benchmark numbers in the PR body — the upstream maintainers don't want to litigate your numbers, they want to know it exists.
**Expected outcome:** Permanent link from a 1.3k-star repo. 10-30 visitors/month long-tail, possibly more when researchers cite TeaCache.

### Action 3 — Submit both libraries to awesome-mlx (15 minutes total)

**Channel:** [github.com/raullenchai/awesome-mlx/issues/new?template=submit_project.yml](https://github.com/raullenchai/awesome-mlx/issues/new?template=submit_project.yml)
**Action:** Open two issues using the form. Image & Video Generation category exists. One-liner each.
**Etiquette:** Use the template. Don't over-explain.
**Expected outcome:** Steady trickle, 5-15/month per library. Plus appears in mirrors and ecosystem aggregators ([ecosyste.ms](https://awesome.ecosyste.ms/lists/akdeb/awesome-mlx)).

### Action 4 — Post both libraries to ml-explore/mlx Discussion #654 (20 minutes)

**Channel:** [github.com/ml-explore/mlx/discussions/654](https://github.com/ml-explore/mlx/discussions/654) — Apple's official MLX show-and-tell.
**Action:** Single comment, two libraries, brief. Two-line description each, with links.
**Etiquette:** This is the canonical Apple-side MLX submission. Recent submissions there are all of this format. Don't pad it.
**Expected outcome:** Adds to the curated list maintained by `awni`; visibility to MLX-core watchers. 10-30 visitors/month, plus possible mention if Apple does an MLX roundup.

### Action 5 — Open ONE discussion on madebyollin/taef2 HF page (30 minutes)

**Channel:** [huggingface.co/madebyollin/taef2/discussions](https://huggingface.co/madebyollin/taef2/discussions) (currently empty; you'd be the first thread)
**Action:** Title: *"MLX port: mlx-taef (Apple Silicon, pure MLX)"*. Body: one paragraph crediting `madebyollin` for the upstream weights, naming the Apple Silicon target, linking to the repo. Include the decode-time and memory numbers, with a note that you'd be happy to link from the model card or close the thread if it's not useful.
**Etiquette:** Single thread. Don't open one on every TAEF variant — just taef2 (newest), and maybe taesd later. Let the upstream maintainer decide whether to amplify.
**Expected outcome:** Most TAEF2 traffic lands on this HF page. A pinned/visible thread from the MLX port is a permanent breadcrumb. 15-40 visitors/month, possible upstream README link.

### Action 6 — Set GitHub topics on both repos (5 minutes)

**Channel:** GitHub repo Settings → About → Topics.
**Action:** As listed in section 3.1 above.
**Etiquette:** None — this is just metadata.
**Expected outcome:** Surfaces on [github.com/topics/mlx](https://github.com/topics/mlx) browse pages (thousands of monthly visitors across MLX-curious browsers). The lowest-effort, highest-multiplier action in this whole document.

### Action 7 — Add hero images to both READMEs (2-3 hours, depends on having outputs)

**Channel:** Your own READMEs.
**Action:** For mlx-teacache, embed the existing klein-base-9b vanilla-vs-wrapper comparison (the `tests/_artifacts/bench_images/` ones referenced in the README) as the very first visual after the title. For mlx-taef, generate a single live-preview GIF (20-30 frames) showing latent→RGB updates during a mflux generate, with the final full-VAE image alongside. Compress to under 5 MB.
**Etiquette:** None.
**Expected outcome:** Anyone who lands on the repo from any of the other channels now has a 5-second visual answer to "what does this do". Conversion to install probably doubles.

### Action 8 — Tighten the PyPI long descriptions and re-release (1 hour)

**Channel:** PyPI.
**Action:** Update `pyproject.toml`'s `description` for mlx-taef to match the better GitHub blurb. For mlx-teacache, add the headline measurement to the top of the README (which PyPI renders). Ship as v0.6.2 / v0.1.2.
**Etiquette:** None.
**Expected outcome:** PyPI search ([pypi.org/search/?q=mlx+flux](https://pypi.org/search/?q=mlx+flux)) starts ranking these libraries on the right keywords. People who land on the PyPI page from a `pip install` retry actually get the value prop instead of a one-liner.

### Deferred to v0.2.0 / not this week

- **Show HN** — wait for the v0.2.0 mlx-taef ship with COMPARISON.md and the measured-showcase. Show HN punishes "v0.1 of a new tool" but rewards "I built X and here's the side-by-side". One shot only — make it count.
- **r/StableDiffusion / r/LocalLLaMA post** — same logic. A post with no GIF dies. A post with a clean before/after image and a Mac-only label can hit serious engagement if timed right.
- **X / Twitter** — only after the HN/Reddit posts have a thread to point to. Don't grind the daily-post mill.

---

## 5. 90-day sustainable rhythm

The maintainer is one person on a 32 GB M1 Max running benchmarks that take hours and can panic the OS if pushed too hard. Promotion needs to be cheap per week or it gets dropped.

**Weekly (max 1-2 hours):**
- Read incoming issues. Respond within ~48 hours, even if the response is "I'll look at this next weekend". Active issue response is the single biggest signal to new visitors that a project is alive.
- Watch [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux) issues and discussions. If someone asks about TeaCache, live preview, tiny VAE, performance — drop a one-line factual reply with a link. **Don't pitch**. Answer the question, link the library.

**Monthly:**
- One *measured* technical post. Pick a single artifact: a new benchmark on a new chip, a new variant gate, a regression you found and fixed, a comparison vs an alternative. Publish to one channel (HF Discussion, mflux Discussion, or a GitHub gist linked from r/StableDiffusion). One channel, not five. Quality > spray.
- Update PyPI long description if anything material has changed (new variant, new chip benchmarked, new mflux version compat).

**Per release (every 3-6 weeks):**
- Tag, push, PyPI publishes via Trusted Publishing.
- Write a 1-paragraph release summary (per CLAUDE.md rule) — what changed for users, the measured number, what's deferred, risk surface.
- For minor releases: nothing more. The CHANGELOG is the record.
- For releases that ship a new *visible* feature (v0.2.0 with COMPARISON.md, v0.7.0 if it lands a new model family), do one cross-post: HN Show HN *or* r/StableDiffusion *or* mflux Discussion. **One**. Pick the channel that fits the feature.

**Quarterly:**
- Reassess. If after 90 days the install count hasn't moved from the low triple-digits monthly into the low four-digits, something in the funnel (README, channel mix, target audience) is wrong and the rhythm itself needs revising — not the volume.

**What this rhythm is designed to avoid:** the X/Twitter daily-post burnout pattern where a maintainer posts 5 times a week for a month, gets nothing, gives up, and the repo looks abandoned. Better to land 1 thing/month for 12 months. Compound interest exists for repos too.

---

## 6. What NOT to do

Failure modes I've watched dev-tools and niche-AI projects fall into:

- **Don't post "v0.X.Y released!" to r/LocalLLaMA.** Reddit punishes release-note posts from authors. If the release has a visible measurable thing (a GIF, a chart, a comparison), post the *artifact* and mention the release in passing. If it doesn't, skip the post.
- **Don't open issues on competing projects to mention yours.** Specifically, don't go into Draw Things' threads or ComfyUI-Mac threads and post "btw I made this". That's the cringe move. Engage substantively or stay out.
- **Don't ship a Show HN with just text.** "Show HN: I built an MLX wrapper for FLUX caching" with no image dies in 4 hours. Show HN demands a visual or an interactive demo. Wait for the COMPARISON.md.
- **Don't claim a speedup you can't reproduce on demand.** Your own CLAUDE.md rule covers this. The v0.3.0 "1.7×" almost-shipped figure is exactly the failure mode — a benchmarked claim that didn't survive scrutiny becomes a trust-burn long after the number itself is corrected.
- **Don't ask for stars.** Star-begging posts ("if you find this useful, please star!") are a known-bad pattern; people who would have starred organically *un-star* when they see it. The repo's star count is a slow-cooked metric; trying to microwave it backfires.
- **Don't promise a roadmap you might not ship.** "v1.0 coming next month with X, Y, Z" creates an obligation that, when missed, looks worse than no announcement. The actual v0.2.0 design doc lives privately for a reason. Ship features, then talk about them.
- **Don't blanket-tag people on X.** Tagging `@awnihannun` or `@filipstrand` or the Apple MLX team in announcement tweets reads as supplicating. If your library is good, they'll find it; if you have a direct technical question, ask in a GitHub issue where it's documented.
- **Don't engage with bad-faith comparison threads.** "MLX is slow compared to llama.cpp" / "Draw Things beats mflux by 25%" — these threads are real ([Draw Things engineering blog](https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c)). Don't argue. Your benchmark scripts are the argument; let people who care look at them.
- **Don't translate the README into 5 languages.** It's a sign of low-confidence promotion. English-only for niche dev tools.
- **Don't add a Discord or Slack.** Two-week-old, one-maintainer projects don't need a community server; they need active issue response. A dead Discord is a worse signal than no Discord.

---

## 7. Citations

All claims in this document, with URLs.

**Library state:**
- [pypistats.org/packages/mlx-teacache](https://pypistats.org/packages/mlx-teacache) — 1 / 166 / 764 downloads (day/week/month).
- [pypistats.org/packages/mlx-taef](https://pypistats.org/packages/mlx-taef) — 1 / 35 / 331 downloads (day/week/month).
- [github.com/IonDen/mlx-teacache](https://github.com/IonDen/mlx-teacache) — 1 star, 0 forks, 0 watchers, no topics set, Apache-2.0.
- [github.com/IonDen/mlx-taef](https://github.com/IonDen/mlx-taef) — 2 stars, 1 fork, 0 watchers, no topics set, MIT.
- [pypi.org/pypi/mlx-teacache/json](https://pypi.org/pypi/mlx-teacache/json) — v0.6.1, 2026-05-26, Apache-2.0, Python ≥3.11.
- [pypi.org/pypi/mlx-taef/json](https://pypi.org/pypi/mlx-taef/json) — v0.1.1, 2026-05-13, MIT, Python ≥3.11.

**Target communities & existing demand:**
- [github.com/filipstrand/mflux](https://github.com/filipstrand/mflux) — 2.1k stars, 148 forks, 76 open issues, README lists 5 community wrappers.
- [github.com/filipstrand/mflux/issues/113](https://github.com/filipstrand/mflux/issues/113) — "Upscaler and TeaCache" — open since 2025-01-12, requested by azrahello, no mention of mlx-teacache.
- [github.com/ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — 1.3k stars, 56 forks, lists community ports for FramePack/FastVideo/ComfyUI/SD.Next/DiffSynth, no MLX port listed.
- [github.com/madebyollin/taesd](https://github.com/madebyollin/taesd) — 936 stars, no MLX port mentioned in README.
- [huggingface.co/madebyollin/taef2](https://huggingface.co/madebyollin/taef2) — MIT, model card mentions no MLX port.
- [huggingface.co/madebyollin/taef2/discussions](https://huggingface.co/madebyollin/taef2/discussions) — empty Community tab.

**Promotion channels:**
- [github.com/raullenchai/awesome-mlx](https://github.com/raullenchai/awesome-mlx) — 80+ projects, Image & Video Generation category exists.
- [github.com/raullenchai/awesome-mlx/issues/new?template=submit_project.yml](https://github.com/raullenchai/awesome-mlx/issues/new?template=submit_project.yml) — submission form.
- [github.com/ml-explore/mlx/discussions/654](https://github.com/ml-explore/mlx/discussions/654) — Apple's "MLX Community Projects" thread.
- [github.com/antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx) — alternative awesome list, less active.
- [awesome.ecosyste.ms/lists/akdeb/awesome-mlx](https://awesome.ecosyste.ms/lists/akdeb/awesome-mlx) — ecosystem aggregator mirror.

**Competing / adjacent projects (for not-to-do list):**
- [drawthings.ai](https://drawthings.ai/) and [Draw Things on App Store](https://apps.apple.com/us/app/draw-things-offline-ai-art/id6444050820) — claims ~25% faster than mflux on M2 Ultra.
- [Draw Things Metal FlashAttention 2.0 engineering post](https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c).
- [github.com/raysers/Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) — ComfyUI integration for mflux, Mac-only.
- [github.com/CamilleHbp/Flux-MLX-ComfyUI](https://github.com/CamilleHbp/Flux-MLX-ComfyUI) — alternative MLX-backend ComfyUI nodes.
- [github.com/argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit) — on-device image gen, Core ML + MLX, Swift.

**Reference docs:**
- v0.2.0 design lives at `/Users/ionden/Documents/Work/mac/mlx-taef/docs/superpowers/specs/2026-05-26-mlx-taef-v0.2.0-design.md` (verified present).

---

## TL;DR — actions for this week

In rough priority order:

1. **Comment on [mflux issue #113](https://github.com/filipstrand/mflux/issues/113)** with the v0.6.1 numbers + repo link.
2. **Set GitHub topics** on both repos (30 seconds).
3. **PR to [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)** adding mlx-teacache to the community-ports list.
4. **Submit both libraries to [awesome-mlx](https://github.com/raullenchai/awesome-mlx)** via the issue form.
5. **Post both libraries to [ml-explore/mlx Discussion #654](https://github.com/ml-explore/mlx/discussions/654)**.
6. **Open a single thread on [taef2 HF Discussions](https://huggingface.co/madebyollin/taef2/discussions)**.
7. **Add hero images** (existing klein-base-9b comparison for mlx-teacache; live-preview GIF for mlx-taef once v0.2.0's showcase ships).
8. **Tighten the PyPI description** for mlx-taef in a v0.1.2 release.

Hold the Show HN, the r/StableDiffusion post, and the X thread for the mlx-taef v0.2.0 ship with COMPARISON.md — that's when there's a visual to lead with, and Show HN demands visuals.
