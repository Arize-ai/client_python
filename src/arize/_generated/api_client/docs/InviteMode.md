# InviteMode

Controls how the user is invited to the account. - `none` — add the user directly with no invitation email (for SSO-only accounts). - `email_link` — send the user an email with a verification link to complete registration. - `temporary_password` — issue a temporary password returned in the `POST /v2/users` response body; the user must reset it on first login. **Treat this value as a secret** — see `UserCreatedResponse.temporary_password` for security guidance. 

## Enum

* `NONE` (value: `'none'`)

* `EMAIL_LINK` (value: `'email_link'`)

* `TEMPORARY_PASSWORD` (value: `'temporary_password'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


