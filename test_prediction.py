from predict import predict_claim

claims = [
    "India landed Chandrayaan-3 on the Moon’s south pole.",
    "The COVID-19 vaccine contains microchips.",
    "Drinking bleach cures COVID-19.",
    "NASA's Artemis mission orbited the Moon.",
]

for claim in claims:
    label, confidence = predict_claim(claim)
    print(f"📰 Claim: {claim}")
    print(f"🔍 Prediction: {label} ({confidence}%)")
    print("-" * 60)