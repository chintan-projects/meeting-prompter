# Heron Router — Model Card

Synthetic evaluation fixture. Heron is one of three confusable router models.

## Latency

The Heron router serves at a p50 latency of 85 milliseconds on a single
Apple Silicon core. The p99 latency is 170 milliseconds under nominal load.

## Capacity

Heron has 700 million parameters and runs comfortably in 1600 megabytes of
resident memory. It supports a context window of 16000 tokens.

## Pricing

The Heron router is billed at 4 cents per thousand requests on the standard
tier. Volume discounts begin at five million monthly requests.

## Fallback

If Heron is unavailable, traffic fails over to the Osprey router, then to the
static response of last resort.
