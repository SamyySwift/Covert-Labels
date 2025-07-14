import requests
import base64
import os
from datetime import datetime

def test_upload_and_save_modified_images(image_paths, batch_id, product_name="Test Product", sku="TEST001", server_url="http://localhost:3000"):
    """
    Test the upload-labels endpoint and save the modified images with embedded patterns
    
    Args:
        image_paths: List of paths to images to upload
        batch_id: Unique batch identifier
        product_name: Name of the product
        sku: SKU identifier
        server_url: URL of the Flask server
    
    Returns:
        dict: Response from the server and saved file paths
    """
    
    # Prepare files for upload
    files = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            files.append(('images', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))
        else:
            print(f"Warning: Image not found: {image_path}")
    
    if not files:
        return {"error": "No valid image files found"}
    
    # Prepare form data
    data = {
        'batch_id': batch_id,
        'product_name': product_name,
        'sku': sku
    }
    
    try:
        # Make request to upload endpoint
        response = requests.post(f"{server_url}/api/upload-labels", files=files, data=data)
        
        # Close file handles
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            result = response.json()
            
            # Create output directory for modified images
            output_dir = f"modified_images_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = []
            
            # Save each modified image
            if 'modified_images' in result:
                for img_data in result['modified_images']:
                    filename = img_data['filename']
                    base64_data = img_data['modified_image']
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(base64_data)
                    
                    # Save to file
                    output_path = os.path.join(output_dir, f"modified_{filename}")
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    saved_files.append(output_path)
                    print(f"✅ Saved modified image: {output_path}")
            
            return {
                "success": True,
                "server_response": result,
                "saved_files": saved_files,
                "output_directory": output_dir
            }
        
        else:
            return {
                "success": False,
                "error": f"Server returned status {response.status_code}: {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

# Example usage function
def run_test():
    """
    Example of how to use the test function
    """
    # Example image paths (replace with your actual image paths)
    test_images = [
        "./images/sartor_activated_charcoal_face_wash_500ml.png",
        "./images/sartor_instant_hand_sanitizer_30ml.png",
        "./images/sartor_intense_moisturizing_lotion_500ml.png",
        "./images/sartor_aleo_vera_face_cleanser_500ml.png"

    ]
    
    # Test parameters
    batch_id = f"test_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Testing upload endpoint with batch_id: {batch_id}")
    
    # Run the test
    result = test_upload_and_save_modified_images(
        image_paths=test_images,
        batch_id=batch_id,
        product_name="Test Product",
        sku="TEST001"
    )
    
    if result["success"]:
        print("\n✅ Test completed successfully!")
        print(f"Modified images saved in: {result['output_directory']}")
        print(f"Server response: {result['server_response']['message']}")
        print(f"Number of files processed: {len(result['saved_files'])}")
    else:
        print(f"\n❌ Test failed: {result['error']}")
    
    return result

if __name__ == "__main__":
    # Run the test
    run_test()