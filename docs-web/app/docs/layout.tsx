import Footer from "@/components/Footer";
import Navbar from "@/components/Navbar";
import DocsSidebar from "@/components/DocsSidebar";

export default function DocsLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <>
      <Navbar />
      <div className="mx-auto flex max-w-7xl gap-10 px-5 pt-24 pb-16">
        <DocsSidebar />
        <main className="min-w-0 flex-1">{children}</main>
      </div>
      <Footer />
    </>
  );
}
