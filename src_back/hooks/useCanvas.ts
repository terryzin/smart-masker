import { useEffect, useRef, useState } from 'react';
import { Point } from '../services/api';

interface UseCanvasProps {
  imageUrl: string | null;
  onPointAdded?: (point: Point) => void;
}

export const useCanvas = ({ imageUrl, onPointAdded }: UseCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [masks, setMasks] = useState<ImageData[]>([]);

  // Initialize canvas context
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      setContext(ctx);
    }
  }, []);

  // Handle image loading and canvas sizing
  useEffect(() => {
    if (!imageUrl || !context || !canvasRef.current) return;

    const image = new Image();
    image.src = imageUrl;
    imageRef.current = image;

    image.onload = () => {
      const canvas = canvasRef.current!;
      const container = canvas.parentElement!;
      
      // Set canvas size to match container while maintaining aspect ratio
      const containerRatio = container.clientWidth / container.clientHeight;
      const imageRatio = image.width / image.height;
      
      let width, height;
      if (containerRatio > imageRatio) {
        height = container.clientHeight;
        width = height * imageRatio;
      } else {
        width = container.clientWidth;
        height = width / imageRatio;
      }

      canvas.width = width;
      canvas.height = height;

      // Draw image
      context.clearRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);
    };
  }, [imageUrl, context]);

  // Handle canvas click
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!context || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Draw point
    context.beginPath();
    context.arc(x, y, 5, 0, 2 * Math.PI);
    context.fillStyle = 'red';
    context.fill();

    const point = { x: Math.round(x), y: Math.round(y) };
    setPoints([...points, point]);
    onPointAdded?.(point);
  };

  // Add mask to canvas
  const addMask = (maskData: ImageData) => {
    setMasks([...masks, maskData]);
    if (context && imageRef.current) {
      // Redraw image
      context.drawImage(imageRef.current, 0, 0, canvasRef.current!.width, canvasRef.current!.height);
      
      // Draw all masks with transparency
      masks.forEach(mask => {
        context.putImageData(mask, 0, 0);
      });
      context.putImageData(maskData, 0, 0);
    }
  };

  // Clear canvas
  const clearCanvas = () => {
    if (context && imageRef.current && canvasRef.current) {
      context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      context.drawImage(imageRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      setPoints([]);
      setMasks([]);
    }
  };

  return {
    canvasRef,
    handleCanvasClick,
    addMask,
    clearCanvas,
    points,
  };
}; 