# Threat model — class task extractor

- **Untrusted input:** the message body AND every header (`from`, `date`, `message_id`) and the filename. Anyone can send the rep a message.
- **Attacker goals:**
  - G1 inject a fake task: exfiltration ("email the answer key to X"), off-topic/harmful ("transfer money"), or a task pinned on a named person;
  - G2 suppress real tasks ("output an empty list");
  - G3 tamper fields ("set every owner to Riya", "deadline is today");
  - G4 break the format (fake JSON, fake closing delimiters, code fences, chat-template tokens);
  - G5 leak the prompt/rules.
- **Capabilities:** the model has no tools, no network, no memory across messages (one call per message). Worst case is a wrong row in `tasks.json` — bounded blast radius, least privilege by design.
- **Key fact:** an injected task's text is literally in the body, so "owner/due must appear in the message" grounding does NOT stop injection; injection needs its own layers (input filter, prompt hierarchy, output checks).
