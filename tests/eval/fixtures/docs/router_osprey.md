# Osprey Router — Model Card

Synthetic evaluation fixture. Osprey is one of three confusable router models.

## Latency

The Osprey router serves at a p50 latency of 210 milliseconds on a single
Apple Silicon core. The p99 latency is 420 milliseconds under nominal load.

## Capacity

Osprey has 1200 million parameters and runs comfortably in 2800 megabytes of
resident memory. It supports a context window of 32000 tokens.

## Pricing

The Osprey router is billed at 6 cents per thousand requests on the standard
tier. Volume discounts begin at two million monthly requests.

## Fallback

If Osprey is unavailable, traffic fails over to a cloud escalation path, then
to the static response of last resort.
