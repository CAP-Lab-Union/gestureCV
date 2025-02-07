export interface GestureData {
    hand: string;
    gesture: string;
    confidence: number;
    landmarks: Array<number>;
  }
  
  export interface GestureResponse {
    gestures: GestureData[];
    frame_data: string;
  }