import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Button,
  Typography,
  styled,
  CircularProgress,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SaveIcon from '@mui/icons-material/Save';
import { useCanvas } from './hooks/useCanvas';
import api, { Point } from './services/api';

const ImageContainer = styled(Paper)(({ theme }) => ({
  width: '100%',
  height: '600px',
  position: 'relative',
  overflow: 'hidden',
  '& canvas': {
    maxWidth: '100%',
    maxHeight: '100%',
  }
}));

interface MaskObject {
  id: string;
  name: string;
  filename: string;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [masks, setMasks] = useState<MaskObject[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentImageFile, setCurrentImageFile] = useState<File | null>(null);

  const { canvasRef, handleCanvasClick, addMask, clearCanvas } = useCanvas({
    imageUrl: selectedImage,
    onPointAdded: handlePointAdded,
  });

  async function handleImageUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      setLoading(true);
      try {
        // Upload image to backend
        const response = await api.uploadImage(file);
        
        // Create object URL for preview
        const imageUrl = URL.createObjectURL(file);
        setSelectedImage(imageUrl);
        setCurrentImageFile(file);
        
        // Clear existing masks
        clearCanvas();
        setMasks([]);
      } catch (error) {
        console.error('Error uploading image:', error);
        // TODO: Show error message to user
      } finally {
        setLoading(false);
      }
    }
  }

  async function handlePointAdded(point: Point) {
    if (!currentImageFile) return;

    setLoading(true);
    try {
      const response = await api.generateMask(currentImageFile.name, [point]);
      
      // Create new mask object
      const newMask: MaskObject = {
        id: `mask-${masks.length + 1}`,
        name: `Object ${masks.length + 1}`,
        filename: response.mask_filename,
      };
      
      setMasks([...masks, newMask]);
      
      // TODO: Convert mask response to ImageData and add to canvas
      // const maskData = await convertMaskToImageData(response.mask_filename);
      // addMask(maskData);
    } catch (error) {
      console.error('Error generating mask:', error);
      // TODO: Show error message to user
    } finally {
      setLoading(false);
    }
  }

  async function handleExportMask() {
    if (!currentImageFile || masks.length === 0) return;

    setLoading(true);
    try {
      // Download the latest mask
      const latestMask = masks[masks.length - 1];
      const maskBlob = await api.downloadMask(latestMask.filename);
      
      // Create download link
      const url = URL.createObjectURL(maskBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${currentImageFile.name.split('.')[0]}_mask.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error exporting mask:', error);
      // TODO: Show error message to user
    } finally {
      setLoading(false);
    }
  }

  function handleDeleteMask(maskId: string) {
    setMasks(masks.filter(mask => mask.id !== maskId));
    // TODO: Remove mask from canvas
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box display="flex" gap={2}>
        {/* Left Panel - Mask List */}
        <Paper sx={{ width: 300, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Mask Objects
          </Typography>
          <List>
            {masks.map((mask) => (
              <ListItem
                key={mask.id}
                secondaryAction={
                  <IconButton edge="end" onClick={() => handleDeleteMask(mask.id)}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemText primary={mask.name} />
              </ListItem>
            ))}
          </List>
        </Paper>

        {/* Right Panel - Image Preview */}
        <Box flex={1}>
          <Box mb={2} display="flex" gap={2}>
            <Button
              variant="contained"
              component="label"
              startIcon={<CloudUploadIcon />}
              disabled={loading}
            >
              Select Image
              <input
                type="file"
                hidden
                accept="image/*"
                onChange={handleImageUpload}
              />
            </Button>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleExportMask}
              disabled={loading || !selectedImage || masks.length === 0}
            >
              Export Mask
            </Button>
          </Box>

          <ImageContainer>
            {loading && (
              <Box
                position="absolute"
                top={0}
                left={0}
                right={0}
                bottom={0}
                display="flex"
                alignItems="center"
                justifyContent="center"
                bgcolor="rgba(255, 255, 255, 0.8)"
                zIndex={1}
              >
                <CircularProgress />
              </Box>
            )}
            {selectedImage ? (
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                style={{ width: '100%', height: '100%' }}
              />
            ) : (
              <Box
                display="flex"
                alignItems="center"
                justifyContent="center"
                height="100%"
              >
                <Typography variant="body1" color="text.secondary">
                  Select an image to begin
                </Typography>
              </Box>
            )}
          </ImageContainer>
        </Box>
      </Box>
    </Container>
  );
}

export default App; 