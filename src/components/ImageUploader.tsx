import React, { useRef, useState } from "react";
// @ts-ignore
import * as ONNX_WEBGPU from "onnxruntime-web/webgpu";
import * as tf from "@tensorflow/tfjs";
import "../App.css";

type ImageUploaderProps = {
  onImageProcessed: (embeddings: any, imageData: ImageData | undefined) => void;
  onStatusChange: (message: string) => void;
};

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageProcessed,
  onStatusChange,
}) => {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (FileReader && files && files.length) {
      const fileReader = new FileReader();
      fileReader.onload = () => {
        const img = imageRef.current;
        if (img) {
          img.onload = () => handleImage(img);
          img.src = fileReader.result as string;
        }
      };
      fileReader.readAsDataURL(files[0]);
    }
  };

  const handleImage = async (img: HTMLImageElement) => {
    onStatusChange(
      `Uploaded image is ${img.width}x${img.height}px. Loading the encoder model (~28 MB).`
    );
    setIsLoading(true);
    ONNX_WEBGPU.env.wasm.numThreads = 1;
    const resizedTensor = await ONNX_WEBGPU.Tensor.fromImage(img, {
      resizedWidth: 1024,
      resizedHeight: 684,
    });
    const resizeImage = resizedTensor.toImageData();
    const imageDataTensor = await ONNX_WEBGPU.Tensor.fromImage(resizeImage);

    let tf_tensor = tf.tensor(
      imageDataTensor.data,
      imageDataTensor.dims as [number, number, number]
    );
    tf_tensor = tf_tensor.reshape([3, 684, 1024]);
    tf_tensor = tf_tensor.transpose([1, 2, 0]).mul(255);

    ONNX_WEBGPU.env.wasm.numThreads = 1;
    // any Blob that contains a valid ORT model would work
    // I'm using Xenova/multilingual-e5-small/onnx/model_quantized.with_runtime_opt.ort
    const response = await fetch("models/mobilesam.encoder.onnx");
    const arrayBuffer = await (await response.blob()).arrayBuffer();
    const session = await ONNX_WEBGPU.InferenceSession.create(arrayBuffer, {
      executionProviders: ["webgpu"],
    });
    const feeds = {
      input_image: new ONNX_WEBGPU.Tensor(
        tf_tensor.dataSync(),
        tf_tensor.shape
      ),
    };
    const start = Date.now();
    try {
      const results = await session.run(feeds);
      const imageData = imageDataTensor.toImageData();
      onImageProcessed(results.image_embeddings, imageData);
    } catch (error) {
      console.log(`caught error: ${error}`);
      onStatusChange(`Error: ${error}`);
    } finally {
      setIsLoading(false);
    }
    const end = Date.now();
    const time_taken = (end - start) / 1000;
    onStatusChange(
      `Embedding generated in ${time_taken} seconds. Click on the image to generate a mask.`
    );
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <img ref={imageRef} style={{ display: "none" }} alt="Uploaded" />
      {isLoading && (
        <div className="spinner">
          <div className="loader"></div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
