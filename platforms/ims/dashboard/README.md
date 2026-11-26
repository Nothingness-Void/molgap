# Always-on IMS dashboard

The supplied services run the IMS dashboard from a Linux host. The collector uses the server's SSH identity to read scheduler and V4 status. The web service reads the resulting file. Choose 中文 or EN in the header.

## Personal profile

Copy [profile.example.json](profile.example.json) to `/etc/ims-dashboard/profile.json`. Put the key file next to it or enter its absolute path. Copy [options.example](options.example) to `/etc/ims-dashboard/options` for systemd to load the profile.

| Setup | `user` | `key` | `remote_root` |
| --- | --- | --- | --- |
| Shared account and shared key | Shared login | Shared key file | Your assigned folder below the shared login home |
| Shared account and personal key | Shared login | Your key file | Your assigned folder below the shared login home |
| Personal account and key | Your login | Your key file | Your account home or a folder below it |

Use the resolved `/lustre/home/users/<user>/...` path. The jobs list matches the folder in scheduler Workdir; submit jobs from that folder or below it. The progress view reads `molgap-v4-submit/runs` inside that folder when present. The Queue H totals, account quotas and group points retain the scheduler's scope. The configured polling interval is 60–120 seconds (default 90); `host` and `ssh_port` are optional. If changing the profile's account or folder, stop both services and move the previous snapshot aside before starting. The files and credentials for one Windows or Linux installation can be set up independently.

## Install on the selected host after approval

Use Linux with systemd, OpenSSH client and Python 3.10+ at /usr/bin/python3. IMS needs bash, realpath, python3 and its scheduler utilities. Create a locked system account and group named ims-dashboard. Copy `check_queue.py` and `dashboard/en.json` with that relative layout into the versioned release directory under /opt/ims-dashboard/releases, then link /opt/ims-dashboard/current to it. Create /etc/ims-dashboard mode 0750 root:ims-dashboard. Provision the profile and chosen key there (key mode 0600, owned by ims-dashboard). Install the verified IMS host key in /etc/ssh/ssh_known_hosts. Use ims-dashboard-web with group ims-dashboard for the reader. An existing installation without the options file continues to use its original account and directory defaults.

Install both adjacent .service files into /etc/systemd/system. Check them with `systemd-analyze verify /etc/systemd/system/ims-dashboard-*.service`, then `systemctl daemon-reload` and `systemctl enable --now ims-dashboard-web ims-dashboard-collector`. Only the collector starts SSH. One query times out after 45 seconds. After success it waits 90 seconds, after failure it doubles the delay up to 900 seconds. The reader marks data older than 240 seconds stale even if the collector stops. Confirm `ss -ltn` shows 127.0.0.1:8765 and open the page. Confirm that pressing Refresh reads the existing snapshot and shows its age. An upgrade from the reduced format needs one successful collection before the new page has data.

## Configure access before routing

Create a [self-hosted Access application](https://developers.cloudflare.com/cloudflare-one/setup/secure-private-apps/private-web-app/) for the full exact hostname. Allow only the named email using the approved identity provider. Review all matching policies, including paths. Exclude bypass and broad allow rules. Only after verifying the policy, configure the named tunnel from cloudflared.yml.example, replace both UUIDs and hostname, validate with `cloudflared tunnel ingress validate`, install the connector as a system service and create its hostname route. Keep the final 404 ingress. Never route the Windows live endpoint. For private devices, [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) with restrictive tailnet grants can instead proxy `tailscale serve --bg 8765`.

For Agent access, first demonstrate the actual client can send both custom headers to a controlled local test endpoint with dummy values and keep secrets out of logs. If supported, protect the exact /api/status path with its own Access application. Include the browser email policy there and a Service Auth rule for one expiring [service token](https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/). Store the credentials in the client secret store. Send CF-Access-Client-Id and CF-Access-Client-Secret on each JSON GET. Leave machine access disabled if the client cannot do this.

## Acceptance and rollback

1. With Windows powered off, use a phone on mobile data. In a private browser verify both / and /api/status redirect to login or deny access without returning dashboard data. A different identity must be denied. Sign in as the permitted viewer and verify last-success time advances across collections.
2. Check that your job names and progress appear, that jobs in a sibling folder are absent, and that the first cards show free A100s and queued jobs. Switch to EN to verify the language file. Multiple refreshes should leave the same observation time until the next scheduled collection.
3. Temporarily block only the collector host’s IMS egress. Confirm the last successful rows and timestamp remain and the dashboard reports stale with an SSH fault. Restore egress and wait through backoff. Restart services and verify recovery from disk.
4. For upgrades, stage another versioned release, verify checksum and offline test first. Keep the previous release, unit files and status.json. Stop both services, switch current to the reviewed release, install matching units, daemon-reload and restart. Repeat acceptance. To roll back, stop services, restore the previous symlink, units and snapshot, reload and start.
5. **Agent visual self-check (UI / copy changes):** before marking a dashboard UI or translation edit done, serve a local snapshot on `127.0.0.1`, open the page in a browser at desktop (~1280px) and mobile (~390px) viewports, toggle 中文/EN, and inspect rendered screenshots for clipping, overflow, stale badges, and readability.
