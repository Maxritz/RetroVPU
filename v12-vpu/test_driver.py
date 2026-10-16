from v4_runtime.phi3_forward import Phi3ForwardDriver
driver = Phi3ForwardDriver('c:\\models\\Phi-3-mini-4k-instruct-q4.gguf', method='v12-vulkan')
print('Driver OK')
print('vk pipelines:', list(driver.pipelines.keys()))
driver.close()
