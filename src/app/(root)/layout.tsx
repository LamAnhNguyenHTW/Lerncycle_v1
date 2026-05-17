/**
 * Minimal root layout for the bare `/` route group.
 * The only page inside is a server-side redirect, so no styling, fonts,
 * or providers are needed — but App Router still requires `<html>`/`<body>`.
 */
export default function RootRedirectLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}
