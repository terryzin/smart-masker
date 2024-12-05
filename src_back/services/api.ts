import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export interface Point {
  x: number;
  y: number;
}

export interface MaskResponse {
  mask_filename: string;
}

const api = {
  uploadImage: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  generateMask: async (filename: string, points: Point[]) => {
    const response = await axios.post<MaskResponse>(`${API_BASE_URL}/generate-mask`, {
      filename,
      points,
    });
    return response.data;
  },

  downloadMask: async (filename: string) => {
    const response = await axios.get(`${API_BASE_URL}/download-mask/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

export default api; 