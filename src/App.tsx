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
  const [highResFeats, setHighResFeats] = useState<any>(null);

  const onImageProcessed = (params: {
    image_embed: any;
    high_res_feats_0: any;
    high_res_feats_1: any;
    imageData: ImageData | undefined;
  }) => {
    setImageEmbeddings({
      image_embed: params.image_embed,
    });
    setHighResFeats({
      high_res_feats_0: params.high_res_feats_0,
      high_res_feats_1: params.high_res_feats_1,
    });
    setImageImageData(params.imageData);
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
        highResFeats={highResFeats}
        onStatusChange={setStatusMessage}
      />
      <div id="status">{statusMessage}</div>
    </>
  );
}

export default App;
