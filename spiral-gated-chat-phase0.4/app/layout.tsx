import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Spiral Gated Chat (Phase0)",
  description: "Probe→state→Main の自己ゲート付きチャット（Phase0）",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body style={{ margin: 0, fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
