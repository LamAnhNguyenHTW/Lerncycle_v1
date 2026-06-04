import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ['pdfjs-dist'],
  experimental: {
    serverActions: {
      bodySizeLimit: '30mb',
    },
    proxyClientMaxBodySize: '30mb',
  },
};

export default nextConfig;
