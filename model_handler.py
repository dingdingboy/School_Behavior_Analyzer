import threading
import time

class ModelHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28
        self._lock = threading.Lock()
        
    def load_model(self, quan_model_path, device, status_callback=None):
        """Load the model in a separate thread"""
        def _load():
            try:
                if status_callback:
                    status_callback("Loading model...")
                    print("Loading model...")
                
                # Import heavy modules only when needed
                from optimum.intel.openvino import OVModelForVisualCausalLM
                from transformers import AutoProcessor
                
                st = time.perf_counter()
                with self._lock:
                    print(f"Loading model from {quan_model_path} on {device}...")
                    self.model = OVModelForVisualCausalLM.from_pretrained(quan_model_path, device=device)
                    self.processor = AutoProcessor.from_pretrained(
                        quan_model_path, 
                        use_fast=True,  
                        min_pixels=self.min_pixels, 
                        max_pixels=self.max_pixels
                    )
                et = time.perf_counter()
                loading_time = et - st
                    
                if status_callback:
                    status_callback(f"Successfully loaded model in {loading_time:.2f} seconds!")
                    print(f"Successfully loaded model in {loading_time:.2f} seconds!")
            except Exception as e:
                if status_callback:
                    print(f"Error loading model: {str(e)}")
                    status_callback(f"Error loading model: {str(e)}")
                
        return threading.Thread(target=_load, daemon=True)

    def run_inference(self, video_path, prompt, max_new_tokens=2048):
        """Run inference with the loaded model"""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Please load the model first.")
            
        # Import heavy modules only when needed
        from qwen_vl_utils import process_vision_info
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"video": video_path, "total_pixels": self.max_pixels, "min_pixels": self.min_pixels},
                ]
            },
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        fps_inputs = video_kwargs['fps']
        
        print("video input:", video_inputs[0].shape)
        num_frames, _, resized_height, resized_width = video_inputs[0].shape
        token_num = int(num_frames * resized_height / 28 * resized_width / 28)

        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            fps=fps_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        st = time.perf_counter()
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        et = time.perf_counter()
        inference_time = et - st
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return output_text[0], inference_time, token_num
