import Link from "next/link";
import { FlaskConical } from "lucide-react";

interface FooterSection {
  title: string;
  links: { label: string; href: string }[];
}

const sections: FooterSection[] = [
  {
    title: "Product",
    links: [
      { label: "AI Chat", href: "/chat" },
      { label: "Paper Search", href: "/search" },
      { label: "Upload Papers", href: "/upload" },
      { label: "Dashboard", href: "/dashboard" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Documentation", href: "/docs" },
      { label: "API Reference", href: "/api/docs" },
    ],
  },
];

export function LandingFooter() {
  return (
    <footer className="border-t bg-muted/30 px-4 py-12">
      <div className="mx-auto grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {/* Branding */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5 text-primary" />
            <span className="text-lg font-bold">PaperAlchemy</span>
          </div>
          <p className="text-sm text-muted-foreground">
            AI-powered research assistant for academic papers.
          </p>
        </div>

        {/* Link sections */}
        {sections.map((section) => (
          <div key={section.title} className="flex flex-col gap-3">
            <h3 className="text-sm font-semibold">{section.title}</h3>
            <ul className="flex flex-col gap-2">
              {section.links.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}

        {/* Connect */}
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold">Connect</h3>
          <ul className="flex flex-col gap-2">
            <li>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                GitHub
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="mx-auto mt-8 max-w-6xl border-t pt-6 text-center text-sm text-muted-foreground">
        &copy; {new Date().getFullYear()} PaperAlchemy. All rights reserved.
      </div>
    </footer>
  );
}
