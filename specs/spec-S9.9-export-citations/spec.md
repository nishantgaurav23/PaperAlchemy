# Spec S9.9 -- Export & Citations

## Overview
Export functionality for papers in multiple formats: BibTeX, Markdown, and presentation slide snippets. Users can export single papers from the paper detail view, or bulk-export papers from collections. All exports include proper citation formatting with arXiv links. Includes copy-to-clipboard and file download capabilities.

## Dependencies
- S9.6 (Paper detail view) — done

## Target Location
`frontend/src/components/export/`

## Functional Requirements

### FR-1: BibTeX Export
- **What**: Generate valid BibTeX entries for papers
- **Inputs**: `Paper` or `PaperDetail` object
- **Outputs**: BibTeX string (e.g., `@article{...}`)
- **Format**: Standard `@article` with fields: author, title, journal (arXiv preprint), year, eprint, archivePrefix, primaryClass, url
- **Edge cases**: Missing authors (use "Unknown"), missing arxiv_id, special characters in title (escape `{`, `}`, `&`, `%`)

### FR-2: Markdown Export
- **What**: Generate Markdown-formatted citation for papers
- **Inputs**: `Paper` or `PaperDetail` object
- **Outputs**: Markdown string with title, authors, year, abstract, arXiv link
- **Format**: Heading with title, metadata block, abstract, link
- **Edge cases**: Missing abstract (omit section), very long author lists (include all)

### FR-3: Slide Snippet Export
- **What**: Generate a concise slide-ready text snippet for presentations
- **Inputs**: `Paper` or `PaperDetail` object
- **Outputs**: Compact text with title, authors (truncated to 3 + "et al."), key point (first sentence of abstract), arXiv link
- **Edge cases**: No abstract available, single author

### FR-4: Copy to Clipboard
- **What**: Copy exported text to clipboard with visual feedback
- **Inputs**: Export format string
- **Outputs**: Clipboard content set, toast/feedback shown
- **Edge cases**: Clipboard API not available (fallback to textarea copy)

### FR-5: File Download
- **What**: Download exported content as a file
- **Inputs**: Export format string, filename
- **Outputs**: Browser triggers file download (.bib, .md, .txt)
- **Edge cases**: Large content for bulk exports

### FR-6: Bulk Export from Collections
- **What**: Export all papers in a collection in a chosen format
- **Inputs**: Collection (array of papers), format choice
- **Outputs**: Combined export (e.g., multi-entry BibTeX file, Markdown list)
- **Edge cases**: Empty collection, collection with 100+ papers

### FR-7: Export UI Components
- **What**: Export button/dropdown on paper detail page and collection pages
- **Inputs**: User interaction (button click, format selection)
- **Outputs**: Dropdown menu with format options, triggers copy or download
- **UI**: Dropdown with icons for each format, "Copy" and "Download" sub-actions

## Tangible Outcomes
- [ ] `formatBibtex(paper)` returns valid BibTeX string
- [ ] `formatMarkdown(paper)` returns formatted Markdown string
- [ ] `formatSlideSnippet(paper)` returns concise slide text
- [ ] `copyToClipboard(text)` copies text and shows feedback
- [ ] `downloadFile(content, filename)` triggers browser download
- [ ] `ExportButton` component renders dropdown with format options
- [ ] `ExportButton` integrates into paper detail page header
- [ ] Bulk export works from collection pages
- [ ] All exports include arXiv URL
- [ ] Special characters in BibTeX are properly escaped
- [ ] All components pass Vitest tests

## Test-Driven Requirements

### Tests to Write First
1. `test_formatBibtex_valid_paper`: Validates BibTeX output format and required fields
2. `test_formatBibtex_special_characters`: Validates escaping of `&`, `%`, `{`, `}`
3. `test_formatBibtex_missing_fields`: Handles missing arxiv_id, authors gracefully
4. `test_formatMarkdown_complete_paper`: Validates Markdown output structure
5. `test_formatMarkdown_no_abstract`: Omits abstract section when missing
6. `test_formatSlideSnippet_truncates_authors`: Shows max 3 authors + "et al."
7. `test_formatSlideSnippet_single_author`: No "et al." for single author
8. `test_copyToClipboard_success`: Verifies clipboard API called
9. `test_downloadFile_creates_blob`: Verifies blob URL and download trigger
10. `test_ExportButton_renders_formats`: Dropdown shows BibTeX, Markdown, Slide options
11. `test_ExportButton_copy_action`: Clicking copy triggers clipboard
12. `test_ExportButton_download_action`: Clicking download triggers file save
13. `test_bulkExport_bibtex`: Multiple papers combined in single BibTeX output
14. `test_bulkExport_empty_collection`: Returns empty string for empty collection

### Mocking Strategy
- Mock `navigator.clipboard.writeText` for clipboard tests
- Mock `URL.createObjectURL` and `document.createElement('a')` for download tests
- Use mock Paper objects with known field values
- No external API calls needed (all client-side formatting)

### Coverage
- All public functions tested
- Edge cases: missing fields, special chars, empty inputs
- UI: render, click interactions, feedback display
