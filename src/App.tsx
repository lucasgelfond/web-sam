import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import ImageCanvas from "./components/ImageCanvas";
import "./App.css";

function App() {
  const [imageEmbeddings, setImageEmbeddings] = useState<any>(null);
  const [imageImageData, setImageImageData] = useState<ImageData | undefined>(
    undefined
  );
  const [statusMessage, setStatusMessage] = useState<string>("");

  const onImageProcessed = (embeddings: any, data: ImageData | undefined) => {
    setImageEmbeddings(embeddings);
    setImageImageData(data);
  };
  const isUsingMobileSam = false;

  return (
    <>
      <ImageUploader
        onImageProcessed={onImageProcessed}
        onStatusChange={setStatusMessage}
        isUsingMobileSam={isUsingMobileSam}
      />
      <ImageCanvas
        isUsingMobileSam={isUsingMobileSam}
        imageEmbeddings={imageEmbeddings}
        imageImageData={imageImageData}
        onStatusChange={setStatusMessage}
      />
      <div id="status">{statusMessage}</div>
    </>
  );
}

export default App;
