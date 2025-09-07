import gradio as gr
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
        return False
    except:
        return False

def generate_marketing_content(prompt, max_length, temperature):
    """Generate marketing content using the API"""
    
    if not prompt.strip():
        return "❌ Please enter a prompt!", "", 0.0
    
    # Check API health first
    if not check_api_health():
        return "❌ API is not running! Please start your FastAPI server first.", "", 0.0
    
    try:
        # Prepare request data
        data = {
            "prompt": prompt,
            "max_length": int(max_length),
            "temperature": float(temperature)
        }
        
        # Make API request
        response = requests.post(f"{API_URL}/generate", json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", "No content generated")
            generation_time = result.get("generation_time", 0.0)
            
            # Format the success message
            status_msg = f"✅ Generated successfully in {generation_time:.2f}s"
            
            return content, status_msg, generation_time
            
        else:
            error_msg = f"❌ API Error ({response.status_code}): {response.text}"
            return error_msg, "Failed", 0.0
            
    except requests.exceptions.Timeout:
        return "❌ Request timed out! Generation took too long.", "Timeout", 0.0
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API! Make sure your FastAPI server is running.", "Connection Error", 0.0
    except Exception as e:
        return f"❌ Error: {str(e)}", "Error", 0.0

def get_example_prompts():
    """Get example prompts from the API"""
    try:
        response = requests.get(f"{API_URL}/examples", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("examples", [])
        return []
    except:
        return [
            "Write marketing content about: AI-powered customer service solutions",
            "Create a social media campaign for a new productivity app",
            "Write an email marketing sequence for Black Friday sales"
        ]

def load_example(example_text):
    """Load an example into the prompt textbox"""
    return example_text

# Create Gradio interface
def create_marketing_app():
    """Create the Gradio marketing content generator app"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .output-text {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    """
    
    with gr.Blocks(css=css, title="Marketing Content Generator", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.Markdown(
            """
            # 🚀 Marketing Content Generator
            Generate engaging marketing content using AI. Powered by fine-tuned Llama 3.1 8B.
            """
        )
        
        # API Status
        with gr.Row():
            api_status = gr.Markdown("🔄 Checking API status...")
        
        # Main interface
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")
                
                prompt_input = gr.Textbox(
                    label="Marketing Prompt",
                    placeholder="Write marketing content about: [your topic here]",
                    lines=3,
                    value="Write marketing content about: AI-powered customer service solutions"
                )
                
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=300,
                        step=50,
                        label="Max Length"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (Creativity)"
                    )
                
                generate_btn = gr.Button("🎯 Generate Content", variant="primary", size="lg")
                
                # Example prompts
                gr.Markdown("### 💡 Example Prompts")
                example_prompts = get_example_prompts()
                
                for i, example in enumerate(example_prompts[:3]):  # Show first 3 examples
                    example_btn = gr.Button(f"📌 {example[:50]}...", size="sm")
                    example_btn.click(
                        fn=lambda x=example: x,
                        outputs=prompt_input
                    )
            
            # Right column - Output
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Generated Content")
                
                content_output = gr.Textbox(
                    label="Marketing Content",
                    lines=15,
                    elem_classes=["output-text"],
                    show_copy_button=True
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False
                )
                
                # Generation info
                with gr.Row():
                    generation_time = gr.Number(
                        label="Generation Time (seconds)",
                        interactive=False
                    )
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown(
                """
                1. **Enter your prompt**: Describe what marketing content you want to generate
                2. **Adjust parameters**:
                   - **Max Length**: How long the generated content should be
                   - **Temperature**: Higher = more creative, Lower = more focused
                3. **Click Generate**: Wait for the AI to create your content
                4. **Copy & Use**: Use the copy button to save your generated content
                
                **Tips**:
                - Be specific in your prompts for better results
                - Try different temperature values for varied outputs
                - Use the example prompts as inspiration
                """
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_marketing_content,
            inputs=[prompt_input, max_length, temperature],
            outputs=[content_output, status_output, generation_time]
        )
        
        # Check API status on load
        def update_api_status():
            if check_api_health():
                return "🟢 **API Status**: Connected and ready!"
            else:
                return "🔴 **API Status**: Not connected. Please start your FastAPI server first."
        
        app.load(fn=update_api_status, outputs=api_status)
    
    return app

# Run the app
if __name__ == "__main__":
    print("🚀 Starting Marketing Content Generator UI...")
    print("📡 Connecting to API at:", API_URL)
    
    # Create and launch the app
    app = create_marketing_app()
    
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Gradio default port
        share=False,            # Set to True for public link
        show_error=True
    )