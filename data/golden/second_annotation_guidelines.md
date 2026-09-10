# Independent Second-Annotation Guidelines

This task measures agreement between two human annotators. Do not look at or infer the first annotator's labels. Label only from the customer message and available thread context.

For each record in `data/golden/second_annotation.jsonl`, independently complete:

- `second_intent`
- `second_action`
- `second_escalation_reason`
- `second_annotator_notes` (optional but encouraged for ambiguity)

Use exactly one of the following intent values:

- `delivery_issue`: A parcel is late, missing, wrongly marked delivered, or has a delivery problem.
- `order_issue`: An order's status, contents, cancellation, or order-specific detail is the central issue.
- `shipping_issue`: Shipping method, carrier, tracking, dispatch, delivery destination, or promised shipping speed is the central issue.
- `payment_issue`: A charge, payment method, billing, declined payment, or payment processing issue.
- `refund_issue`: A refund, returned money, refund status, or request for money back.
- `account_issue`: Account access, password, sign-in, account settings, closure, or account status.
- `prime_video_issue`: Prime Video playback, streaming, video availability, or video quality.
- `device_issue`: Amazon hardware such as Kindle, Echo, Fire TV, Fire Stick, or device functionality.
- `product_or_content_issue`: A product, listing, content item, movie, show, or item-quality issue not better covered above.
- `support_complaint`: Dissatisfaction with customer service, previous support interactions, or unresolved service treatment.
- `other`: A greeting, reaction, unrelated message, or a message without enough information for another intent.

Choose one action:

- `auto_handle`: Safe to resolve using the available historical evidence without human intervention.
- `escalate`: Requires human review because of sensitive action, ambiguity, insufficient information, complaint, financial/account action, or an unresolved/complex issue.

For `second_escalation_reason`, write a short plain-language reason. For `auto_handle`, write `none` if no escalation is needed. Do not use the first annotator's labels, the production agent, keyword rules, or automated suggestions to complete these fields.

When finished, run:

```bash
python scripts/verify_second_annotation.py
```

Agreement is calculated only after all 40 records are complete.
