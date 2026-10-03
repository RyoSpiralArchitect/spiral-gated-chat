import { getProviderConfig } from "@/lib/providers";

export const dynamic = "force-dynamic";
export function GET() {
  return Response.json(getProviderConfig());
}
