export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center gap-8 py-16">
      <div className="flex flex-col items-center gap-4">
        <h1 className="text-4xl font-bold tracking-tight">PaperAlchemy</h1>
        <p className="max-w-md text-center text-lg text-muted-foreground">
          AI Research Assistant — search, chat, and explore academic papers with
          citation-backed answers.
        </p>
      </div>
    </div>
  );
}
