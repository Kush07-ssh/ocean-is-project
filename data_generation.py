import json
import random


def generate_ocean_dataset(num_rows=50):
    dataset = []

    # Templates for professional analysis
    traits_desc = {
        "Openness": ["creative and intellectually curious", "prefers routine and literal thinking",
                     "balanced between innovation and tradition"],
        "Conscientiousness": ["highly organized and reliable", "spontaneous and potentially disorganized",
                              "dependable in most tasks"],
        "Extraversion": ["outgoing and energetic", "reserved and quiet", "socially adaptable"],
        "Agreeableness": ["trusting and cooperative", "competitive and skeptical", "selective in social harmony"],
        "Neuroticism": ["emotionally sensitive", "exceptionally resilient and calm", "stable with occasional stress"]
    }

    for i in range(num_rows):
        # Generate random scores based on item counts in PDF
        scores = {
            "Extraversion": random.randint(8, 40),
            "Agreeableness": random.randint(9, 45),
            "Conscientiousness": random.randint(9, 45),
            "Neuroticism": random.randint(8, 40),
            "Openness": random.randint(10, 50)
        }

        # Build Analysis Logic
        analysis = []
        recommendations = []

        if scores["Openness"] > 40:
            analysis.append(
                "You possess a high degree of intellectual curiosity and value aesthetic experiences[cite: 38, 59].")
            recommendations.append(
                "Seek roles that require 'out-of-the-box' thinking or artistic innovation[cite: 13, 33].")
        elif scores["Openness"] < 20:
            analysis.append("You prefer practical, routine-based work over abstract theory[cite: 43, 49].")
            recommendations.append(
                "Focus on specialized technical fields where consistency is valued over constant change.")

        if scores["Conscientiousness"] > 40:
            analysis.append(
                "Your profile indicates you are a reliable worker who finishes tasks despite obstacles[cite: 21, 36].")
            recommendations.append(
                "You are well-suited for project management or roles requiring high precision[cite: 11, 41].")

        if scores["Neuroticism"] < 15:
            analysis.append("You handle stress exceptionally well and remain calm in tense situations[cite: 17, 42].")
            recommendations.append("Consider high-pressure environments like emergency response or crisis management.")

        if scores["Extraversion"] > 35:
            analysis.append(
                "You are someone who generates a lot of enthusiasm and enjoys leading groups[cite: 24, 34].")
            recommendations.append("Leverage your sociability in leadership or client-facing positions[cite: 44].")

        full_output = " ".join(analysis) + " Recommendations: " + " ".join(recommendations)

        # Format for Fine-Tuning
        entry = {
            "instruction": "Generate a professional personality report based on the provided OCEAN scores.",
            "input": f"Scores - {scores}",
            "output": full_output
        }
        dataset.append(entry)

    return dataset


# Save to JSON
ocean_data = generate_ocean_dataset(150)
with open('ocean_finetuning_data.json', 'w') as f:
    json.dump(ocean_data, f, indent=2)

print("Dataset of 150 rows generated successfully.")