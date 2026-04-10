
import requests
import time
import os
import json
import yaml
import sys

# Add current dir to sys.path to import model_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_manager import VLMModelManager

def test_models():
    manager = VLMModelManager()
    # Disable pre-start to manage manually
    manager.stop_all()
    
    results = {}
    
    for model_name in list(manager.models.keys()):
        print(f"\n>>> Testing {model_name}...")
        results[model_name] = {}
        
        # Test Reasoning ON
        print(f"  Enabling Reasoning...")
        success = manager.start_server(model_name, reasoning="on")
        if not success:
            results[model_name]["reasoning_on"] = "FAILED_TO_START"
            print(f"  [!] Failed to start with reasoning on")
        else:
            time.sleep(2)
            resp = manager.query(model_name, "Solve 123 * 456 step by step.", max_tokens=256, reasoning="on")
            if resp["status"] == "success":
                has_cot = "<details>" in resp["text"]
                results[model_name]["reasoning_on"] = "WORKS" if has_cot else "NO_COT_OUTPUT"
                print(f"  Reasoning ON -> {results[model_name]['reasoning_on']}")
            else:
                results[model_name]["reasoning_on"] = f"QUERY_ERROR: {resp['message']}"
                print(f"  Reasoning ON -> ERROR: {resp['message']}")

        # Test Reasoning OFF
        print(f"  Disabling Reasoning...")
        manager.stop_server(model_name)
        success = manager.start_server(model_name, reasoning="off")
        if not success:
            results[model_name]["reasoning_off"] = "FAILED_TO_START"
            print(f"  [!] Failed to start with reasoning off")
        else:
            time.sleep(2)
            resp = manager.query(model_name, "Hello, who are you?", max_tokens=50, reasoning="off")
            results[model_name]["reasoning_off"] = "WORKS" if resp["status"] == "success" else f"QUERY_ERROR: {resp['message']}"
            print(f"  Reasoning OFF -> {results[model_name]['reasoning_off']}")
        
        manager.stop_server(model_name)

    print("\n\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(json.dumps(results, indent=2))
    
    with open("reasoning_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_models()
