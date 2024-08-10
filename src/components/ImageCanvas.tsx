import React, { useCallback, useRef, useEffect } from "react";
// @ts-ignore
import * as ONNX_WEBGPU from "onnxruntime-web/webgpu";

type ImageCanvasProps = {
  imageEmbeddings: any;
  imageImageData: ImageData | undefined;
  highResFeats: any;
  onStatusChange: (message: string) => void;
  isUsingMobileSam?: boolean;
};

const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageEmbeddings,
  imageImageData,
  highResFeats,
  onStatusChange,
  isUsingMobileSam,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleClick = useCallback(
    async (event: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !imageImageData || !imageEmbeddings) return;
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

      const preparePoints = (coords: number[], labels: number[]) => {
        const inputPointCoords = new Float32Array([coords[0], coords[1], 0, 0]);
        const inputPointLabels = new Float32Array([labels[0], -1]);

        // Normalize coordinates
        inputPointCoords[0] = (inputPointCoords[0] / canvas.width) * 1024;
        inputPointCoords[1] = (inputPointCoords[1] / canvas.height) * 1024;

        return { inputPointCoords, inputPointLabels };
      };

      const { inputPointCoords, inputPointLabels } = preparePoints([x, y], [1]);

      const pointCoords = new ONNX_WEBGPU.Tensor(inputPointCoords, [1, 2, 2]);
      const pointLabels = new ONNX_WEBGPU.Tensor(inputPointLabels, [1, 2]);
      const maskInput = new ONNX_WEBGPU.Tensor(
        new Float32Array(256 * 256),
        [1, 1, 256, 256]
      );
      const hasMask = new ONNX_WEBGPU.Tensor(new Float32Array([0]), [1]);
      const originalImageSize = new ONNX_WEBGPU.Tensor(
        new Float32Array([canvas.height, canvas.width]),
        [2]
      );

      const url = isUsingMobileSam
        ? "https://sam2-download.b-cdn.net/models/mobilesam.decoder.quant.onnx"
        : "https://sam2-download.b-cdn.net/sam2_hiera_tiny.decoder.onnx";
      // Fetch the decoder model
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

      const arrayBuffer = await response.arrayBuffer();

      // Create the decoding session using the downloaded model data
      const decodingSession = await ONNX_WEBGPU.InferenceSession.create(
        arrayBuffer,
        {
          executionProviders: ["webgpu"],
        }
      );

      const prepareInputs = (
        imageEmbed: ONNX_WEBGPU.Tensor,
        highResFeats0: ONNX_WEBGPU.Tensor,
        highResFeats1: ONNX_WEBGPU.Tensor,
        pointCoords: ONNX_WEBGPU.Tensor,
        pointLabels: ONNX_WEBGPU.Tensor
      ) => {
        const maskInput = new ONNX_WEBGPU.Tensor(
          new Float32Array(256 * 256),
          [1, 1, 256, 256]
        );
        const hasMaskInput = new ONNX_WEBGPU.Tensor(new Float32Array([0]), [1]);

        return {
          image_embed: imageEmbed,
          high_res_feats_0: highResFeats0,
          high_res_feats_1: highResFeats1,
          point_coords: pointCoords,
          point_labels: pointLabels,
          mask_input: maskInput,
          has_mask_input: hasMaskInput,
        };
      };

      const decodingFeeds = {
        image_embed: new ONNX_WEBGPU.Tensor(
          new Float32Array(imageEmbeddings.image_embed.data),
          imageEmbeddings.image_embed.dims
        ),
        high_res_feats_0: new ONNX_WEBGPU.Tensor(
          new Float32Array(highResFeats.high_res_feats_0.data),
          highResFeats.high_res_feats_0.dims
        ),
        high_res_feats_1: new ONNX_WEBGPU.Tensor(
          new Float32Array(highResFeats.high_res_feats_1.data),
          highResFeats.high_res_feats_1.dims
        ),
        point_coords: pointCoords,
        point_labels: pointLabels,
        mask_input: new ONNX_WEBGPU.Tensor(
          new Float32Array(256 * 256),
          [1, 1, 256, 256]
        ),
        has_mask_input: new ONNX_WEBGPU.Tensor(new Float32Array([0]), [1]),
      };

      const start = Date.now();
      try {
        // Run inference
        const results = await decodingSession.run(decodingFeeds, {
          masks: true,
          iou_predictions: true,
        });
        console.log("Decoding results", results);
        const { masks, iou_predictions } = results;
        console.log({ masks, iou_predictions });
        // Process the output
        const maskImageData = masks.toImageData();
        context.globalAlpha = 0.5;
        // Convert image data to image bitmap and resize to original image size
        let imageBitmap = await createImageBitmap(maskImageData);
        context.drawImage(imageBitmap, 0, 0, canvas.width, canvas.height);
      } catch (error) {
        console.log(`caught error: ${error}`);
        onStatusChange(`Error: ${error}`);
      }
      const end = Date.now();
      console.log(`generating masks took ${(end - start) / 1000} seconds`);
      onStatusChange(
        `Mask generated. Click on the image to generate a new mask.`
      );
    },
    [
      imageEmbeddings,
      imageImageData,
      highResFeats,
      onStatusChange,
      isUsingMobileSam,
    ]
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
