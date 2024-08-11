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

    // Create a canvas element to resize the image
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext("2d");

    if (ctx) {
      // Calculate the scaling factor to fit the image within 1024x1024
      const scale = Math.max(1024 / img.width, 1024 / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;

      // Calculate position to center the image
      const x = (1024 - scaledWidth) / 2;
      const y = (1024 - scaledHeight) / 2;

      // Fill the canvas with white background
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, 1024, 1024);

      // Draw the image onto the canvas, resizing it and adding space
      ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

      //   // Convert canvas to blob
      //   canvas.toBlob((blob) => {
      //     if (blob) {
      //       // Create a download link
      //       const url = URL.createObjectURL(blob);
      //       const link = document.createElement("a");
      //       link.href = url;
      //       link.download = "resized_image.png";
      //       document.body.appendChild(link);
      //       link.click();
      //       document.body.removeChild(link);
      //       URL.revokeObjectURL(url);
      //     }
      //   }, "image/png");

      // Get the image data from the canvas
      // @ts-ignore
      const imageData = ctx.getImageData(0, 0, 1024, 1024);
      const rgbData = [];

      // Define mean and std for normalization
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];

      // Remove alpha channel, flatten the data, and normalize
      for (let i = 0; i < imageData.data.length; i += 4) {
        for (let j = 0; j < 3; j++) {
          const pixelValue = imageData.data[i + j] / 255.0;
          const normalizedValue = (pixelValue - mean[j]) / std[j];
          rgbData.push(normalizedValue);
        }
        // Alpha channel (imageData.data[i + 3]) is discarded
      }

      // Create a tensor with shape [1024, 1024, 3]
      const tensor = tf.tensor3d(rgbData, [1024, 1024, 3]);

      // Transpose and reshape to [1, 3, 1024, 1024]
      const batchedTensor = tf.tidy(() => {
        const transposed = tf.transpose(tensor, [2, 0, 1]);
        return tf.expandDims(transposed, 0);
      });

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
        image: new ONNX_WEBGPU.Tensor(
          batchedTensor.dataSync(),
          batchedTensor.shape
        ),
      };
      const start = Date.now();
      try {
        const results = await session.run(feeds);

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
          // @ts-ignore
          imageData: ctx.getImageData(
            0,
            0,
            ctx.canvas.width,
            ctx.canvas.height
          ),
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
    }
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
