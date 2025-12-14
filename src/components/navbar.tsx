import Link from 'next/link';
import { Github } from 'lucide-react';

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-semibold hover:opacity-80 transition-opacity">
          <span className="text-lg">deep-label</span>
        </Link>
        <nav className="flex items-center gap-2">
          <a
            href="https://github.com/derekphilipau/deep-label"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-8 items-center gap-2 whitespace-nowrap rounded-md border border-input bg-background px-3 text-xs font-medium hover:bg-accent hover:text-accent-foreground transition-colors"
          >
            <Github className="h-4 w-4 shrink-0" />
            View on GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
