# I fine-tuned a local AI on merger law

The standalone article is [reports/maud-fine-tuning.html](../../reports/maud-fine-tuning.html). Open it directly in a browser. CSS, JavaScript, the responsive validation figure and downloadable evidence are embedded; no server, external font or model is required. [maud-blog.zip](../../reports/maud-blog.zip) includes the article, standalone SVG figures and evidence files.

The design follows the original project engineering article’s paper/rust palette, typography and annotated figures. The article’s diagrams and experiment graphics are generated for this project.

Edit `article.template.html`, `article.css` and `article.js` here, then rebuild:

```bash
python3.12 -m venv .operator/blog-venv
.operator/blog-venv/bin/pip install -r writing/maud-blog/requirements.txt
.operator/blog-venv/bin/python scripts/build_maud_blog.py
```

The builder reads the frozen final-test report, validation summary and selection manifest. It checks the selected experiment and editorial rare-answer counts; it does not train, infer, rescore, change the test protocol or contact APIs. Matplotlib generates standalone, accessible-by-caption SVG plots with embedded glyph paths. Fixed plot IDs and archive timestamps make builds reproducible with the pinned dependencies.

The article uses a measured first-person technical case-study structure. A results summary, descriptive headings, labeled findings and a controlled weighting comparison make it easy to skim. It covers the local RTX 3090, why MAUD, published methods versus our adaptations, six normal epochs, the weighted branch, final results and untested next hypotheses. It distinguishes final test from validation, shows the majority baseline, explains both rare-answer denominators and limits the claim to predefined MAUD questions. Expandable configuration details and downloadable evidence retain the methods and provenance.

The builder enforces a 3,000-word maximum and records the count in `reports/maud-blog-assets/build.json`. It counts all HTML body text, including closed details, tables and image alt text, while excluding styles and scripts. Reading time uses that rendered count. The current article has 2,545 words. Future ideas are explicitly unrun; the completed final test remains closed to tuning.

Browser review checks desktop and 390/320-pixel layouts, summary and weighting-table values, interactive metrics, figure enlargement and Escape dismissal, internal links, embedded downloads, images, no-JavaScript reading and absence of runtime network requests. The verification receipt is retained in `docs/publication-checks.json`; preview screenshots are local operator artifacts. Publication to a public website is outside this HTML artifact change.
