# InviteMode

Controls how the user is invited to the account. - `NONE` — add the user directly with no invitation email (for SSO-only accounts). - `EMAIL_LINK` — send the user an email with a verification link to complete registration. - `TEMPORARY_PASSWORD` — issue a temporary password returned in the `POST /v2/users` response body; the user must reset it on first login. **Treat this value as a secret** — see `CreateUserResponse.temporary_password` for security guidance. 

## Enum

* `NONE` (value: `'NONE'`)

* `EMAIL_LINK` (value: `'EMAIL_LINK'`)

* `TEMPORARY_PASSWORD` (value: `'TEMPORARY_PASSWORD'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


