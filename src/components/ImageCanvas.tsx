import React, { useCallback, useRef, useEffect, useState } from "react";
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
  const [maskThreshold, setMaskThreshold] = useState<number>(2);

  const handleClick = useCallback(
    async (event: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !imageImageData || !imageEmbeddings) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      console.log("Clicked position:", x, y);
      onStatusChange(
        `Clicked on (${x}, ${y}). Downloading the decoder model if needed and generating masks...`
      );

      let context = canvas.getContext("2d");
      if (!context) return;
      context.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = imageImageData.width;
      canvas.height = imageImageData.height;
      context.putImageData(imageImageData, 0, 0);
      context.fillStyle = "rgba(0, 0, 139, 0.7)"; // Dark blue with some transparency
      context.fillRect(x - 1, y - 1, 2, 2); // Smaller 2x2 pixel

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
        : "https://sam2-download.b-cdn.net/sam2_hiera_small.decoder.onnx";

      const decodingSession = await fetchAndCreateSession(url);

      console.log(imageImageData.width, imageImageData.height);

      const decodingFeeds = prepareDecodingInputs(
        imageEmbeddings,
        highResFeats,
        pointCoords,
        pointLabels
      );

      const start = Date.now();
      try {
        const results = await runInference(decodingSession, decodingFeeds);
        console.log("Decoding results", results);
        const { masks, iou_predictions } = results;
        console.log({ masks, iou_predictions });

        const { maskWidth, maskHeight, numMasks } = getMaskDimensions(masks);
        console.log({ maskWidth, maskHeight, numMasks });

        // Draw all three masks
        const colors = [
          [0, 0, 139],
          [0, 0, 139],
          [0, 0, 139],
        ];
        for (let i = 0; i < numMasks; i++) {
          const mask = selectMask(masks, i);
          let imageData = context.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
          );
          imageData = drawMask(
            imageData,
            mask,
            colors[i] as [number, number, number],
            0.3,
            maskWidth,
            maskHeight,
            maskThreshold
          );
          context.putImageData(imageData, 0, 0);

          drawContour(
            context,
            mask,
            maskWidth,
            maskHeight,
            canvas.width,
            canvas.height,
            maskThreshold
          );
        }

        // Post-process masks
        const originalSize = [canvas.height, canvas.width];
        const postProcessedMasks = postProcessMasks(
          masks,
          // @ts-ignore
          originalSize,
          maskThreshold
        );

        // Draw post-processed masks
        for (let i = 0; i < postProcessedMasks.length; i++) {
          let imageData = context.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
          );
          imageData = drawMask(
            imageData,
            postProcessedMasks[i],
            colors[i % colors.length] as [number, number, number],
            0.3,
            canvas.width,
            canvas.height,
            0.5 // Use a fixed threshold for post-processed masks
          );
          context.putImageData(imageData, 0, 0);

          drawContour(
            context,
            postProcessedMasks[i],
            canvas.width,
            canvas.height,
            canvas.width,
            canvas.height,
            0.5 // Use a fixed threshold for post-processed masks
          );
        }
      } catch (error) {
        console.log(`caught error: ${error}`);
        onStatusChange(`Error: ${error}`);
      }
      const end = Date.now();
      console.log(`generating masks took ${(end - start) / 1000} seconds`);
      onStatusChange(
        `Masks generated. Click on the image to generate new masks.`
      );
    },
    [
      imageEmbeddings,
      imageImageData,
      highResFeats,
      onStatusChange,
      isUsingMobileSam,
      maskThreshold,
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

  return (
    <div>
      <canvas ref={canvasRef} />
      <div>
        <label htmlFor="threshold">Mask Threshold: </label>
        <input
          type="range"
          id="threshold"
          min="0"
          max="20"
          step="0.1"
          value={maskThreshold}
          onChange={(e) => setMaskThreshold(parseFloat(e.target.value))}
        />
        <span>{maskThreshold}</span>
      </div>
    </div>
  );
};

async function fetchAndCreateSession(url: string) {
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

  return await ONNX_WEBGPU.InferenceSession.create(arrayBuffer, {
    executionProviders: ["webgpu"],
  });
}

function prepareDecodingInputs(
  imageEmbeddings: any,
  highResFeats: any,
  pointCoords: any,
  pointLabels: any
) {
  return {
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
}

async function runInference(session: any, feeds: any) {
  return await session.run(feeds, {
    masks: true,
    iou_predictions: true,
  });
}

function getMaskDimensions(masks: any) {
  return {
    maskWidth: masks.dims[2],
    maskHeight: masks.dims[3],
    numMasks: masks.dims[1],
  };
}

function selectMask(masks: any, maskIndex: number) {
  const maskData = masks.data;
  const maskWidth = masks.dims[2];
  const maskHeight = masks.dims[3];

  const mask = new Float32Array(maskWidth * maskHeight);
  for (let i = 0; i < maskWidth * maskHeight; i++) {
    mask[i] = maskData[i + maskIndex * maskWidth * maskHeight];
  }

  return mask;
}

function drawMask(
  imageData: ImageData,
  mask: Float32Array,
  color: [number, number, number],
  alpha: number,
  maskWidth: number,
  maskHeight: number,
  threshold: number
) {
  const imageDataCopy = new ImageData(
    new Uint8ClampedArray(imageData.data),
    imageData.width,
    imageData.height
  );
  const scaleX = imageData.width / maskWidth;
  const scaleY = imageData.height / maskHeight;

  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const maskIndex = y * maskWidth + x;
      if (mask[maskIndex] > threshold) {
        const startX = Math.floor(x * scaleX);
        const startY = Math.floor(y * scaleY);
        const endX = Math.floor((x + 1) * scaleX);
        const endY = Math.floor((y + 1) * scaleY);

        for (let py = startY; py < endY; py++) {
          for (let px = startX; px < endX; px++) {
            const index = (py * imageData.width + px) * 4;
            imageDataCopy.data[index] = color[0];
            imageDataCopy.data[index + 1] = color[1];
            imageDataCopy.data[index + 2] = color[2];
          }
        }
      }
    }
  }

  // Blend the mask with the original image
  for (let i = 0; i < imageData.data.length; i += 4) {
    imageData.data[i] =
      (1 - alpha) * imageData.data[i] + alpha * imageDataCopy.data[i];
    imageData.data[i + 1] =
      (1 - alpha) * imageData.data[i + 1] + alpha * imageDataCopy.data[i + 1];
    imageData.data[i + 2] =
      (1 - alpha) * imageData.data[i + 2] + alpha * imageDataCopy.data[i + 2];
  }

  return imageData;
}

function drawContour(
  context: CanvasRenderingContext2D,
  mask: Float32Array,
  maskWidth: number,
  maskHeight: number,
  canvasWidth: number,
  canvasHeight: number,
  threshold: number
) {
  const scaleX = canvasWidth / maskWidth;
  const scaleY = canvasHeight / maskHeight;
  context.beginPath();
  context.strokeStyle = "white";
  context.lineWidth = 2;

  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const i = y * maskWidth + x;
      if (mask[i] > threshold) {
        const hasLowerNeighbor =
          (x > 0 && mask[i - 1] <= threshold) ||
          (x < maskWidth - 1 && mask[i + 1] <= threshold) ||
          (y > 0 && mask[i - maskWidth] <= threshold) ||
          (y < maskHeight - 1 && mask[i + maskWidth] <= threshold);

        if (hasLowerNeighbor) {
          const canvasX = x * scaleX;
          const canvasY = y * scaleY;
          context.moveTo(canvasX, canvasY);
          context.lineTo(canvasX + scaleX, canvasY);
          context.lineTo(canvasX + scaleX, canvasY + scaleY);
          context.lineTo(canvasX, canvasY + scaleY);
          context.lineTo(canvasX, canvasY);
        }
      }
    }
  }
  context.stroke();
}

function postProcessMasks(
  masks: any,
  originalSize: [number, number],
  threshold: number
): Float32Array[] {
  const [height, width] = originalSize;
  const { maskWidth, maskHeight, numMasks } = getMaskDimensions(masks);

  const processedMasks: Float32Array[] = [];

  for (let i = 0; i < numMasks; i++) {
    const mask = selectMask(masks, i);
    const processedMask = new Float32Array(height * width);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const maskX = Math.floor((x * maskWidth) / width);
        const maskY = Math.floor((y * maskHeight) / height);
        const maskIndex = maskY * maskWidth + maskX;
        processedMask[y * width + x] = mask[maskIndex] > threshold ? 1 : 0;
      }
    }

    processedMasks.push(processedMask);
  }

  return processedMasks;
}

export default ImageCanvas;
