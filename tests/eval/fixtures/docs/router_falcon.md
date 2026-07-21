# Falcon Router — Model Card

Synthetic evaluation fixture. Falcon is one of three confusable router models.

## Latency

The Falcon router serves at a p50 latency of 120 milliseconds on a single
Apple Silicon core. The p99 latency is 240 milliseconds under nominal load.

## Capacity

Falcon has 350 million parameters and runs comfortably in 900 megabytes of
resident memory. It supports a context window of 8000 tokens.

## Pricing

The Falcon router is billed at 2 cents per thousand requests on the standard
tier. Volume discounts begin at ten million monthly requests.

## Fallback

If Falcon is unavailable, traffic fails over to the Heron router, then to the
static response of last resort.
