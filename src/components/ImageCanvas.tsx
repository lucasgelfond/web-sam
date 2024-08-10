import React, { useCallback, useRef, useEffect } from "react";
// @ts-ignore
import * as ONNX_WEBGPU from "onnxruntime-web/webgpu";

type ImageCanvasProps = {
  imageEmbeddings: any;
  imageImageData: ImageData | undefined;
  onStatusChange: (message: string) => void;
  isUsingMobileSam?: boolean;
};

const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageEmbeddings,
  imageImageData,
  onStatusChange,
  isUsingMobileSam,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleClick = useCallback(
    async (event: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !imageImageData) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      console.log("Clicked position:", x, y);
      onStatusChange(
        `Clicked on (${x}, ${y}). Downloading the decoder model if needed and generating mask...`
      );

      let context = canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = imageImageData.width;
      canvas.height = imageImageData.height;
      context.putImageData(imageImageData, 0, 0);
      context.fillStyle = "green";
      context.fillRect(x, y, 5, 5);
      const pointCoords = new ONNX_WEBGPU.Tensor(
        new Float32Array([x, y, 0, 0]),
        [1, 2, 2]
      );
      const pointLabels = new ONNX_WEBGPU.Tensor(
        new Float32Array([0, -1]),
        [1, 2]
      );
      const maskInput = new ONNX_WEBGPU.Tensor(
        new Float32Array(256 * 256),
        [1, 1, 256, 256]
      );
      const hasMask = new ONNX_WEBGPU.Tensor(new Float32Array([0]), [1]);
      const originalImageSize = new ONNX_WEBGPU.Tensor(
        new Float32Array([684, 1024]),
        [2]
      );

      const url = isUsingMobileSam
        ? "https://sam2-download.b-cdn.net/models/mobilesam.decoder.quant.onnx"
        : "https://sam2-download.b-cdn.net/sam2_hiera_tiny.decoder.ort";
      // Fetch the encoder model
      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Content-Type": "application/octet-stream",
        },
        mode: "cors",
        credentials: "omit",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get the total size of the file
      const totalSize = Number(response.headers.get("Content-Length"));

      // Create a buffer to store the file contents
      const buffer = new Uint8Array(totalSize);
      let receivedLength = 0;

      // Get the reader from the response body
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get reader for model stream");
      }

      // Read the data in chunks
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer.set(value, receivedLength);
        receivedLength += value.length;

        // You can add a progress indicator here if needed
        // const percentComplete = (receivedLength / totalSize) * 100;
        // console.log(`Downloaded ${percentComplete.toFixed(2)}%`);
      }

      // Create the decoding session using the downloaded model data
      const decodingSession = await ONNX_WEBGPU.InferenceSession.create(
        buffer,
        {
          executionProviders: ["webgpu"],
        }
      );

      // console.log("Decoder session created from stream");
      // const response = await fetch("models/mobilesam.decoder.quant.onnx");
      // const arrayBuffer = await (await response.blob()).arrayBuffer();
      // const decodingSession = await ONNX_WEBGPU.InferenceSession.create(
      //   arrayBuffer,
      //   {
      //     executionProviders: ["webgpu"],
      //   }
      // );
      console.log("Decoder session", decodingSession);
      const decodingFeeds = {
        image_embeddings: imageEmbeddings,
        point_coords: pointCoords,
        point_labels: pointLabels,
        mask_input: maskInput,
        has_mask_input: hasMask,
        orig_im_size: originalImageSize,
      };

      const start = Date.now();
      try {
        const results = await decodingSession.run(decodingFeeds);
        const mask = results.masks;
        const maskImageData = mask.toImageData();
        context.globalAlpha = 0.5;
        // convert image data to image bitmap
        let imageBitmap = await createImageBitmap(maskImageData);
        context.drawImage(imageBitmap, 0, 0);
      } catch (error) {
        console.log(`caught error: ${error}`);
      }
      const end = Date.now();
      console.log(`generating masks took ${(end - start) / 1000} seconds`);
      onStatusChange(
        `Mask generated. Click on the image to generate a new mask.`
      );
    },
    [imageEmbeddings, imageImageData, onStatusChange]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageImageData) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    context.putImageData(imageImageData, 0, 0);
  }, [imageImageData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener("click", handleClick);

    return () => {
      canvas.removeEventListener("click", handleClick);
    };
  }, [handleClick]);

  return <canvas ref={canvasRef} />;
};

export default ImageCanvas;
