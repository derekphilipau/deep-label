import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'deep-label',
  description: 'High-density bounding-box annotations for artworks.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

