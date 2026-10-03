import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "spiral · Gated Chat Lab",
  description: "同じ会話を固定ゲートと自動ゲートで比較。注目点、使われた記憶、視点の変化をたどる実験室。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body>{children}</body></html>;
}
