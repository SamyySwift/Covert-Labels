from openai import OpenAI
import base64
import json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def analyze_covert_labels(image1_path, image2_path):
    """
    Analyze two covert labels with embedded microdots to determine if patterns match
    """
    
    # Read and encode images
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    image1_base64 = encode_image(image1_path)
    image2_base64 = encode_image(image2_path)
    
    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional
            "X-Title": "Covert Label Analyzer",   # Optional
        },
        extra_body={},
        model="meta-llama/llama-4-maverick:free",
        messages=[
            {
                "role": "system",
                "content": """
You are a specialized AI system designed to analyze covert security labels that contain embedded microdot patterns. Your primary task is to compare two label images and determine if their microdot patterns match.

**ANALYSIS CRITERIA:**
- Look for tiny dots, barely visible patterns embedded in the labels
- Compare the geometric arrangement, positioning, and distribution of these microdots
- Account for minor variations due to printing tolerances, lighting conditions, or image quality
- If 20% or more of the pattern elements match between the two images, consider it a MATCH
- Be reasonably tolerant of missing dots, slight positional shifts, or minor distortions
- Focus on the overall pattern structure rather than perfect pixel-by-pixel matching

**RESPONSE FORMAT:**
Respond with ONLY:
- "YES" if the patterns match (≥20% similarity)
- "NO" if the patterns don't match (<20% similarity)

**IMPORTANT:**
- Do not be overly strict - real-world printing and imaging introduces natural variations
- Consider the overall geometric relationship between dots, not just individual dot presence
- Account for rotation, scaling, or slight perspective differences
- Focus on pattern authenticity rather than perfect reproduction
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze these two covert labels and determine if their embedded microdot patterns match. Look carefully for tiny dots or subtle patterns that may be barely visible. Compare the geometric arrangement and respond with YES if they match (≥80% similarity) or NO if they don't match."
                    },
                    {
                        "type": "text",
                        "text": "First label image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image1_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Second label image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image2_base64}"
                        }
                    }
                ]
            }
        ]
    )
    
    response = completion.choices[0].message.content.strip().upper()
    return response == "YES"

# Example usage
if __name__ == "__main__":
    # Replace with actual image paths
    image1 = "clg_output_images/Sartor Activated Charcoal Face Wash 500ml/06250101/Sartor Activated Charcoal Face Wash 500ml_dot_var_25.png"
    image2 = "clg_output_images/Sartor Activated Charcoal Face Wash 500ml/06250101/Sartor Activated Charcoal Face Wash 500ml_dot_var_25.png"
    
    try:
        is_match = analyze_covert_labels(image1, image2)
        print(is_match)
        print(f"Pattern Match: {'YES' if is_match else 'NO'}")
    except Exception as e:
        print(f"Error analyzing labels: {e}")