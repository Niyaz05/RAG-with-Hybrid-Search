# API Reference

## Authentication

All requests to the Aperture Cloud API must include an `Authorization` header
with a bearer token. Tokens are issued from the developer dashboard under
Settings > API Keys and are valid for 90 days before requiring rotation.

Example header:

    Authorization: Bearer ap_live_x7f2k9...

Requests without a valid token return `401 Unauthorized`. Requests with an
expired token return `403 Forbidden` with the error code `TOKEN_EXPIRED`.

## Rate Limiting

All endpoints under `/v1/` are rate limited to 100 requests per minute per
API key. Exceeding this limit returns `429 Too Many Requests` with a
`Retry-After` header indicating the number of seconds to wait.

Enterprise plan customers can request a higher limit of 1000 requests per
minute by contacting the platform team.

## Endpoints

### POST /v1/documents

Uploads a new document for indexing. Accepts `multipart/form-data` with a
`file` field. Maximum file size is 25MB. Returns a `document_id` that can be
used to check indexing status.

### GET /v1/documents/{document_id}/status

Returns the indexing status of a document: `pending`, `processing`,
`indexed`, or `failed`. Failed documents include an `error_code` field.

### POST /v1/query

Submits a question against the indexed corpus. Accepts a JSON body with a
`question` field and an optional `top_k` parameter (default 5). Returns an
`answer` field along with a `citations` array referencing source chunks.

## Error Codes

| Code | Meaning |
|------|---------|
| `TOKEN_EXPIRED` | The bearer token has passed its 90-day validity window |
| `RATE_LIMIT_EXCEEDED` | Too many requests within the rolling 60-second window |
| `DOCUMENT_TOO_LARGE` | Uploaded file exceeds the 25MB limit |
| `UNSUPPORTED_FORMAT` | File extension is not one of pdf, docx, md, txt, html |
