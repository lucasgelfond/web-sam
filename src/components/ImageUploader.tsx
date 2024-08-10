import React, { useRef, useState } from "react";
// @ts-ignore
import * as ONNX_WEBGPU from "onnxruntime-web/webgpu";
import * as tf from "@tensorflow/tfjs";
import "../App.css";

type ImageUploaderProps = {
  onImageProcessed: (params: {
    image_embed: any;
    high_res_feats_0: any;
    high_res_feats_1: any;
    imageData: ImageData | undefined;
  }) => void;
  onStatusChange: (message: string) => void;
  isUsingMobileSam?: boolean;
};

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageProcessed,
  onStatusChange,
  isUsingMobileSam = true,
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
    // ONNX_WEBGPU.env.wasm.numThreads = 1;
    const resizedTensor = await ONNX_WEBGPU.Tensor.fromImage(img, {
      resizedWidth: 1024,
      resizedHeight: 1024,
    });
    const resizeImage = resizedTensor.toImageData();
    const imageDataTensor = await ONNX_WEBGPU.Tensor.fromImage(resizeImage);

    let tf_tensor = tf.tensor(
      imageDataTensor.data,
      imageDataTensor.dims as [number, number, number]
    );
    tf_tensor = tf_tensor.reshape([1, 1024, 1024, 3]);
    tf_tensor = tf_tensor.transpose([0, 3, 1, 2]).mul(255);

    const url = isUsingMobileSam
      ? "https://sam2-download.b-cdn.net/models/mobilesam.encoder.onnx"
      : "https://sam2-download.b-cdn.net/sam2_hiera_small.encoder.with_runtime_opt.ort";
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/octet-stream",
      },
      mode: "cors",
      credentials: "omit",
    });

    // Check if the response is ok
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the total size of the file
    const totalSize = Number(response.headers.get("Content-Length"));

    // Create a new Uint8Array to store the file contents
    const buffer = new Uint8Array(totalSize);
    let receivedLength = 0;

    // Get the reader from the response body
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Failed to get reader for model stream");
    }

    // Read the data
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer.set(value, receivedLength);
      receivedLength += value.length;

      // You can add a progress indicator here if needed
      // const percentComplete = (receivedLength / totalSize) * 100;
      // console.log(`Downloaded ${percentComplete.toFixed(2)}%`);
    }

    // Create a blob from the buffer
    const blob = new Blob([buffer], { type: "application/octet-stream" });

    // Convert blob to ArrayBuffer
    const arrayBuffer = await blob.arrayBuffer();

    // Create the inference session using the downloaded model data
    const session = await ONNX_WEBGPU.InferenceSession.create(arrayBuffer, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "disabled",
    });

    console.log("Session created", session);
    const feeds = {
      image: new ONNX_WEBGPU.Tensor(tf_tensor.dataSync(), tf_tensor.shape),
    };
    const start = Date.now();
    try {
      const results = await session.run(feeds);
      console.log({ results });
      const imageData = imageDataTensor.toImageData();
      console.log({ results });

      // Loop through each result and check for GPU data
      for (const [key, tensor] of Object.entries(results)) {
        if (tensor instanceof ONNX_WEBGPU.Tensor) {
          // @ts-ignore
          const gpuData = tensor?.gpuData;
          if (gpuData) {
            console.log(`${key} has GPU data`);
          } else {
            console.log(`${key} does not have GPU data`);
          }
        } else {
          console.log(`${key} is not an ONNX_WEBGPU.Tensor`);
        }
      }
      onImageProcessed({
        image_embed: results.image_embed,
        high_res_feats_0: results.high_res_feats_0,
        high_res_feats_1: results.high_res_feats_1,
        imageData: imageData,
      });
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
