# Spec S9.5 -- Paper Upload UI

## Overview
Drag-and-drop PDF upload page for the Next.js frontend. Users can upload academic PDFs which get sent to the backend for parsing, indexing, and AI analysis. The page displays upload progress, then shows the paper's AI-generated summary, key highlights, and methodology analysis once processing completes.

This is a **frontend-only spec** — it builds the UI that will call backend upload/analysis endpoints (S8.1-S8.4). Until those backend specs are implemented, the UI uses mock responses for development.

## Dependencies
- S9.2 (Layout & Navigation) — **done** ✅

## Target Location
- `frontend/src/app/upload/page.tsx` — Upload page
- `frontend/src/components/upload/` — Upload components
- `frontend/src/lib/api/upload.ts` — Upload API client
- `frontend/src/types/upload.ts` — Upload TypeScript types

## Functional Requirements

### FR-1: Drag-and-Drop Upload Zone
- **What**: A drop zone that accepts PDF files via drag-and-drop or file picker
- **Inputs**: PDF file (single file upload)
- **Outputs**: File selected state, file metadata display (name, size)
- **Constraints**: PDF-only (reject non-PDF), max 50MB, single file at a time
- **Edge cases**:
  - Non-PDF file dropped → show error message
  - File exceeds 50MB → show error message
  - Multiple files dropped → only accept first PDF
  - Drag enter/leave visual feedback

### FR-2: Upload Progress
- **What**: Visual progress indicator during file upload
- **Inputs**: Upload request in progress
- **Outputs**: Progress bar (0-100%), status text (uploading, processing, complete, error)
- **States**: idle → uploading → processing → complete | error
- **Edge cases**:
  - Network error during upload → show retry button
  - Cancel upload mid-way → reset to idle state

### FR-3: Analysis Results Display
- **What**: After successful upload, display the paper's AI analysis results
- **Inputs**: Backend response with paper metadata + analysis
- **Outputs**: Tabbed/sectioned display showing:
  - **Paper Info**: Title, authors, abstract, categories, arXiv link (if available)
  - **Summary**: AI-generated structured summary (objective, method, findings, contribution, limitations)
  - **Highlights**: Key insights and novel contributions
  - **Methodology**: Methods, datasets, baselines, results
- **Edge cases**:
  - Partial analysis (some sections missing) → show available sections, indicate missing
  - Very long content → scrollable sections

### FR-4: Upload Another / Reset
- **What**: After viewing results, user can upload another paper
- **Inputs**: Click "Upload Another" button
- **Outputs**: Reset to initial drop zone state, clear previous results

### FR-5: Mock API for Development
- **What**: Until backend S8.x specs are implemented, use mock responses
- **Inputs**: File upload request
- **Outputs**: Simulated delay (2s) + mock analysis data
- **Toggle**: Environment variable or feature flag to switch between mock and real API

## Tangible Outcomes
- [ ] Upload page renders at `/upload` route
- [ ] Drag-and-drop zone accepts PDF files
- [ ] File picker fallback works (click to browse)
- [ ] Non-PDF files are rejected with error message
- [ ] Files >50MB are rejected with error message
- [ ] Progress bar shows during upload
- [ ] Analysis results display in tabs (Summary, Highlights, Methodology)
- [ ] "Upload Another" resets the page
- [ ] Mock API returns sample data for development
- [ ] All components have Vitest tests
- [ ] Page is responsive (mobile + desktop)
- [ ] Dark mode works correctly

## Test-Driven Requirements

### Tests to Write First
1. `test_drop_zone_renders`: Drop zone renders with correct prompt text and icon
2. `test_file_picker_accepts_pdf`: File input accepts .pdf files only
3. `test_rejects_non_pdf`: Shows error when non-PDF file is selected
4. `test_rejects_large_file`: Shows error when file exceeds 50MB
5. `test_drag_and_drop_visual_feedback`: Drop zone changes style on drag enter/leave
6. `test_upload_progress_display`: Progress bar renders during upload
7. `test_upload_success_shows_results`: Analysis results display after successful upload
8. `test_analysis_tabs_switch`: Clicking tabs switches displayed content
9. `test_upload_another_resets`: "Upload Another" button resets to initial state
10. `test_upload_error_shows_retry`: Error state shows retry button
11. `test_api_client_sends_formdata`: Upload API sends FormData with file
12. `test_mock_api_returns_data`: Mock API returns sample analysis data

### Mocking Strategy
- Mock `fetch` / API client for upload requests
- Mock file objects with `new File(["content"], "test.pdf", { type: "application/pdf" })`
- Use `fireEvent.drop` for drag-and-drop testing
- Use `userEvent.upload` for file picker testing

### Coverage
- All components tested (DropZone, ProgressBar, AnalysisResults, UploadPage)
- Error states covered (invalid file, network error, oversize)
- User interactions tested (drag, drop, click, tab switch, reset)
