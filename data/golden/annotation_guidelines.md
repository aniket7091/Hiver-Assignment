# AmazonHelp Intent Annotation Guidelines

## Purpose

Each example represents a customer support issue from the
AmazonHelp Twitter conversation dataset.

The goal is to assign the intent that best describes the
customer's primary support problem.

---

## Labels

### 1. delivery_issue

Use when the customer has a problem with receiving a package.

Examples:

- Package has not arrived.
- Package is late.
- Package says delivered but was not received.
- Customer cannot find the delivered package.

Do not use for general questions about shipping speed.

---

### 2. order_issue

Use when the problem is specifically about an Amazon order
or order status and does not primarily concern delivery speed
or payment.

Examples:

- Problem with an order.
- Question about an order.
- Order status problem.
- Customer provides an order number while reporting an
  order-specific problem.

---

### 3. shipping_issue

Use for shipping method, shipping speed, Prime shipping,
or promised delivery-speed problems.

Examples:

- Two-day shipping did not arrive in two days.
- Question about Prime shipping.
- Shipping speed changed.
- Complaint about promised shipping time.

---

### 4. payment_issue

Use for charges, billing, payment methods, or payment
processing.

Examples:

- Charged unexpectedly.
- Payment failed.
- Wrong charge.
- Payment method problem.

---

### 5. refund_issue

Use when the main problem is obtaining or understanding
a refund.

Examples:

- Refund has not arrived.
- Customer asks for money back.
- Refund amount is incorrect.
- Customer asks about refund status.

---

### 6. account_issue

Use for Amazon account access, account settings, account
closure, or account-level problems.

Examples:

- Cannot access account.
- Wants account closed.
- Account settings problem.

---

### 7. prime_video_issue

Use for Prime Video playback, streaming, video errors,
or video quality problems.

Examples:

- Prime Video will not play.
- Video playback error.
- Streaming problem.
- Prime Video quality issue.

---

### 8. device_issue

Use for Amazon hardware or device problems.

Examples:

- Fire TV Stick does not work.
- Echo device problem.
- Kindle problem.
- Amazon device malfunction.

---

### 9. product_or_content_issue

Use for product, movie, show, listing, or content-specific
feedback that does not fit another operational intent.

Examples:

- Audio/video synchronization problem in a movie.
- Incorrect product information.
- Product/content feedback.

---

### 10. support_complaint

Use when the primary issue is dissatisfaction with Amazon's
customer support rather than the underlying product/order
problem.

Examples:

- "Your customer service is terrible."
- "Three agents gave me different answers."
- "Nobody is helping me."

If the customer clearly states an underlying operational
problem, prefer that operational intent when possible.

---

### 11. other

Use when the message cannot reasonably be assigned to one
of the above intents.

Examples:

- "HELP"
- "Thank you"
- "Why?"
- Greetings.
- Very short reactions.
- Unrelated content.
- Insufficient information.