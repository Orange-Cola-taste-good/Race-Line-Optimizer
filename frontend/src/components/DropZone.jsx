import { useDropzone } from 'react-dropzone';

export default function DropZone({ onFile, label, accept = { 'image/*': ['.png', '.jpg', '.jpeg'] } }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: files => files[0] && onFile(files[0]),
    accept,
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      style={{
        border: `2px dashed ${isDragActive ? '#e11d48' : '#333'}`,
        borderRadius: 8,
        padding: '32px 16px',
        textAlign: 'center',
        cursor: 'pointer',
        color: '#666',
        fontSize: 14,
        transition: 'border-color 0.2s',
        background: isDragActive ? '#1a0a0e' : 'transparent',
      }}
    >
      <input {...getInputProps()} />
      <div style={{ fontSize: 28, marginBottom: 8 }}>📁</div>
      {isDragActive ? 'Drop it here…' : label}
    </div>
  );
}
