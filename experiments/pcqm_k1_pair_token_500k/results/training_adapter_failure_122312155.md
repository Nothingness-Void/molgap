# Training adapter failure 122312155

Kunshan job `122312155` exited in the shell adapter before Python started. The
cluster's Bash treats expansion of an empty array as an unbound variable under
`set -u`; preflight mode had not exposed this because its array was nonempty.
The adapter now uses one optional scalar argument. No cache, model, optimizer,
or scientific result was read or modified by this failed job.

