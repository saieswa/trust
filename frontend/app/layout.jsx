import "./globals.css";

export const metadata = {
  title: "Trust-aware RAG",
  description: "Ask questions with evidence from an uploaded document.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}