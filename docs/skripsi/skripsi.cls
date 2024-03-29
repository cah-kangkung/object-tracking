%-------------------------------------------------------------------------------
% Template Skripsi 
% Modifikasi dari JTETI FT UGM
%
% Dibuat oleh @aji purwinarko
% Hasil fork dari file Guntur D Putra UGM
% Silakan digunakan dan diedit seperlunya, semoga bermanfaat
%
% (c) 2019
%-------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
%Created by Pekik Nurwantoro, Universitas Gadjah Mada, Yogyakarta, Indonesia
%Modifikasi by Aji Purwinarko
\ProvidesClass{skripsi}

\RequirePackage[english,bahasa]{babel}
\RequirePackage{indentfirst}
\RequirePackage{setspace}
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{graphicx,latexsym}

\setlength{\paperwidth}{210mm}
\setlength{\paperheight}{297mm}
\usepackage[pass]{geometry}
\usepackage{verbatim,enumerate}

\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{multirow}
\usepackage{float}
\usepackage{indentfirst}

\newif\if@msthesis
\newif\if@msproposal
\DeclareOption{skripsi}{\@msthesistrue\@msproposalfalse}
\DeclareOption{proposal}{\@msthesisfalse\@msproposaltrue}

\newif\if@singlesp
\newif\if@doublesp
\DeclareOption{satuspasi}{\@singlesptrue\@doublespfalse}
\DeclareOption{duaspasi}{\@singlespfalse\@doublesptrue}
\newif\if@onehalfsp
\@onehalfspfalse
\DeclareOption{satusetengahspasi}{\@onehalfsptrue}

\newif\if@langindo
\newif\if@langeng
\DeclareOption{indonesia}{\@langindotrue\@langengfalse}
\DeclareOption{inggris}{\@langindofalse\@langengtrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{skripsi,satusetengahspasi,indonesia}
\ProcessOptions
\LoadClass[a4paper,12pt]{report}


%------------------------------------------------------------
%Layout
%------------------------------------------------------------
\setlength{\topmargin}{-0.9cm}
\setlength{\headheight}{12pt}
\setlength{\headsep}{2.3cm}
\setlength{\topskip}{1ex}
\setlength{\oddsidemargin}{1.46cm}
\setlength{\evensidemargin}{1.46cm}
\setlength{\textwidth}{14.3cm}
\setlength{\textheight}{22cm}
\setlength{\footskip}{1.5cm}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}

\setlength{\parindent}{3em} %Indent approx 6 character
\setlength{\parskip}{1ex} %length between paragraph

%------------------------------------------------------------
%The form of page numbering
%------------------------------------------------------------
\pagestyle{myheadings}
\markright{}

%------------------------------------------------------------
%Define page numbering in the first chapter
%------------------------------------------------------------
\def\ps@chapterheading{%
  \let\@evenhead\@empty\let\@oddhead\@empty
  \def\@oddfoot{\hfil\thepage\hfil}%
  \def\@evenfoot{\hfil\thepage\hfil}
  }

%------------------------------------------------------------
%Redefine chapter and sections
%------------------------------------------------------------
\setcounter{secnumdepth}{2}
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@Roman\c@chapter}
%\renewcommand \thesection {\@arabic\c@section.}
\renewcommand \thesection {\@arabic\c@chapter.\@arabic\c@section}
%\renewcommand\thesubsection {\@alph\c@subsection.}
\renewcommand\thesubsection {\@arabic\c@chapter.\@arabic\c@section.\@arabic\c@subsection}
%\renewcommand\thesubsubsection{\@roman\c@subsubsection.}
%\renewcommand\thesubsubsection{}
\renewcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\renewcommand{\chapter}{\clearpage\thispagestyle{chapterheading}%
  \global\@topnum\z@ %Prevents figures from going at top of page
  \@afterindenttrue %Indent the 1st paragraph
  \secdef\@chapter\@schapter}
\renewcommand{\@makechapterhead}[1]{%
  {\parindent \z@ \centering \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \large\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 5\p@
    \fi
    \interlinepenalty\@M
    \large \bfseries #1\par\nobreak
    \vskip 20\p@
    }}
\renewcommand{\@makeschapterhead}[1]{%
  {\parindent \z@ \centering \normalfont
    \interlinepenalty\@M \large \bfseries #1\par\nobreak \vskip 20\p@ }}
%\renewcommand{\section}{\@startsection {section}{1}{\z@}%
%                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
%                                   {2.3ex \@plus.2ex}%
%                                   {\normalfont\normalsize\bfseries\centering}}
\renewcommand{\section}{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {1ex \@plus.2ex}%
                                   {\normalfont\normalsize\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {.2ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
%\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\parindent}%
%                                    {3.25ex \@plus1ex \@minus.2ex}%
%                                    {-1em}%
%                                    {\normalfont\normalsize\bfseries}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand{\paragraph}{\subparagraph}

\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@equation}
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@figure}
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@table}

%------------------------------------------------------------
%Redefine caption names
%------------------------------------------------------------
\def\captionsbahasa{%
\def\prefacename{KATA PENGANTAR}%
\def\contentsname{DAFTAR ISI}%
\def\listfigurename{DAFTAR GAMBAR}%
\def\listtablename{DAFTAR TABEL}%
\def\listappendixname{DAFTAR LAMPIRAN}%
\def\nomenclaturename{DAFTAR SINGKATAN}%
\def\abstractname{ABSTRAK}%
\def\acknowledgmentname{HALAMAN PERSEMBAHAN}%
\def\statementname{LEMBAR PERNYATAAN}
\def\approvalname{LEMBAR PERSETUJUAN HASIL SIDANG SKRIPSI}
\def\agreementname{LEMBAR PENGESAHAN}
\def\partname{BAGIAN}%
\def\chaptername{BAB}%
\def\appendixname{LAMPIRAN}%
\def\refname{DAFTAR PUSTAKA}%
\def\bibname{DAFTAR PUSTAKA}%
\def\indexname{Indek}%
\def\figurename{Gambar}%
\def\tablename{Tabel}%
\def\pagename{Halaman}%

}

%English
\def\captionsenglish{%
\def\prefacename{PREFACE}%
\def\contentsname{CONTENTS}%
\def\listfigurename{LIST OF FIGURES}%
\def\listtablename{LIST OF TABLES}%
\def\listappendixname{LIST OF APPENDICES}%
\def\nomenclaturename{NOMENCLATURE}%
\def\abstractname{\emph{ABSTRACT}}%
\def\partname{PART}%
\def\chaptername{CHAPTER}%
\def\appendixname{APPENDIX}%
\def\refname{REFERENCES}%
\def\bibname{REFERENCES}%
\def\indexname{Index}%
\def\figurename{Figure}%
\def\tablename{Table}%
\def\pagename{Page}%
}

%------------------------------------------------------------]
%Define thesis's inputs
%------------------------------------------------------------
\gdef\@university{Universitas Negeri Jakarta}
\gdef\@faculty{Fakultas Matematika dan Ilmu Pengetahuan Alam}
\gdef\@city{Jakarta}
\newcommand{\titleind}[1]{\gdef\@titleind{#1}}
\newcommand{\@titleind}{}
\newcommand{\fullname}[1]{\gdef\@fullname{#1}}
\newcommand{\@fullname}{}
\newcommand{\idnum}[1]{\gdef\@idnum{#1}}
\newcommand{\@idnum}{}
\newcommand{\examdate}[1]{\gdef\@examdate{#1}}
\newcommand{\approvaldate}[1]{\gdef\@approvaldate{#1}}
\newcommand{\@examdate}{\number\day~\ifcase\month\or
    Januari\or Februari\or Maret\or April\or Mei\or Juni\or
    Juli\or Agustus\or September\or Oktober\or November\or Desember\fi
    \space \number\year}
\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\@degree}{}
\newcommand{\yearsubmit}[1]{\gdef\@yearsubmit{#1}}
\newcommand{\@yearsubmit}{}
\newcommand{\program}[1]{\gdef\@program{#1}}
\newcommand{\@program}{}
\newcommand{\headprogram}[1]{\gdef\@headprogram{#1}}
\newcommand{\@headprogram}{}
\newcommand{\dept}[1]{\gdef\@dept{#1}}
\newcommand{\@dept}{}
\newcommand{\firstsupervisor}[1]{\gdef\@firstsupervisor{#1}}
\newcommand{\@firstsupervisor}{}
\newcommand{\secondsupervisor}[1]{\gdef\@secondsupervisor{#1}}
\newcommand{\@secondsupervisor}{}
\newcommand{\firstnip}[1]{\gdef\@firstnip{#1}}
\newcommand{\@firstnip}{}
\newcommand{\secondnip}[1]{\gdef\@secondnip{#1}}
\newcommand{\@secondnip}{}
\newcommand{\statementdate}[1]{\gdef\@statementdate{#1}}
\newcommand{\@statementdate}{}
\newcommand{\agreementdate}[1]{\gdef\@agreementdate{#1}}
\newcommand{\@agreementdate}{}
\newcommand{\dean}[1]{\gdef\@dean{#1}}
\newcommand{\@dean}{}
\newcommand{\deannip}[1]{\gdef\@deannip{#1}}
\newcommand{\@deannip}{}
\newcommand{\secretary}[1]{\gdef\@secretary{#1}}
\newcommand{\@secretary}{}
\newcommand{\secretarynip}[1]{\gdef\@secretarynip{#1}}
\newcommand{\@secretarynip}{}
\newcommand{\examinera}[1]{\gdef\@examinera{#1}}
\newcommand{\@examinera}{}
\newcommand{\examinerb}[1]{\gdef\@examinerb{#1}}
\newcommand{\@examinerb}{}
\newcommand{\examinernipa}[1]{\gdef\@examinernipa{#1}}
\newcommand{\@examinernipa}{}
\newcommand{\examinernipb}[1]{\gdef\@examinernipb{#1}}
\newcommand{\@examinernipb}{}

\newcommand{\@listattachmentsname}{DAFTAR LAMPIRAN}

%------------------------------------------------------------
%Define cover in Indonesian
%------------------------------------------------------------
\def\cover{%
  \thispagestyle{empty}%
  \pagenumbering{roman}
  \setcounter{page}{1}
    \begin{center}
      \begin{singlespace}
      \fontsize{16pt}{16pt}\selectfont\MakeUppercase{\normalfont\bfseries\@titleind}\par\nobreak\
      
	   \vspace{1cm}
       {\normalfont\bfseries\fontsize{14pt}{14pt}
              Skripsi\\
              \vspace{1cm}
    		  Disusun untuk memenuhi salah satu syarat\\
    		  memperoleh gelar Sarjana Komputer
          
        }
        \vspace{1.5cm}
        \includegraphics[width=4cm]{image/logo_unj.jpg}\\
        \vspace{1cm}
        {\normalfont\bfseries\fontsize{14pt}{14pt}
              Hafizhun Alim\\
              1313617032
        }
       \vfill
       {\normalfont 
         \fontsize{14pt}{14pt}\MakeUppercase{\normalfont\bfseries Program Studi \@dept}\\
         \fontsize{14pt}{14pt}\MakeUppercase{\normalfont\bfseries\@faculty}\\
          \fontsize{14pt}{14pt}\MakeUppercase{\normalfont\bfseries\@university}\\
          \fontsize{14pt}{14pt}\MakeUppercase{\normalfont\bfseries\@yearsubmit}\\}
      \end{singlespace}
      \end{center}
}



%------------------------------------------------------------
%Define Pernyataan
%------------------------------------------------------------
\def\statementpage{%
  \chapter*{\statementname}%
 Saya menyatakan dengan sesungguhnya bahwa skripsi dengan judul \textbf{"\@titleind"} yang disusun sebagai syarat untuk memperoleh gelar Sarjana Komputer dari Program Studi Ilmu Komputer Universitas Negeri Jakarta adalah karya ilmiah saya dengan arahan dari dosen pembimbing. 
 
 Sumber informasi yang diperoleh dari penulis lain yang telah dipublikasikan dan disebutkan dalam teks skripsi ini, telah dicantumkan dalam Daftar Pustaka sesuai dengan norma, kaidah dan etika penulisan ilmiah.
 
 Jika dikemudian hari ditemukan sebagian besar skripsi ini bukan hasil karya saya sendiri dalam bagian-bagian tertentu, saya bersedia menerima sanksi pencabutan gelar akademik yang saya sanding dan sanksi-sanksi lainnya sesuai dengan peraturan perundang-undangan yang berlaku.

\vspace*{\fill}
    
\begin{flushright}   
    \begin{tabular}{p{7.5cm}l} 
	&\@city, \@statementdate\\	
	&\\
	&\\
	&\\
	&\\
	&\\
	&\@fullname \\
	&\@idnum
	\end{tabular}
\end{flushright}
\addcontentsline{toc}{chapter}{\statementname}
}


%------------------------------------------------------------
%Define Persetujuan
%------------------------------------------------------------
\def\agreementpage{%
\chapter*{\agreementname}%
Dengan ini saya mahasiswa Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Jakarta\\

\begin{tabular}{p{2.7cm} p{0.5mm} p{9.2cm}} 
	Nama 	& : & \@fullname \\
	NIM		& : & \@idnum \\
	Program Studi & : & Ilmu Komputer \\
	Judul	& : & \textit{Fish Movement Tracking} Menggunakan Metode \textit{Gaussian Mixture Models (GMM)} dan \textit{Kalman Filter}
\end{tabular}

\noindent \\Menyatakan bahwa penelitian ini telah siap diajukan untuk sidang skripsi.

\vspace{0.4cm}
\begin{center}
	Menyetujui, \\
 	\vspace{0.25cm}
\end{center}

\begin{singlespace}
 \noindent  \begin{tabular}{@{}lcl}
      Dosen Pembimbing I & \hspace{2.45cm} & Dosen Pembimbing II \\
      & \\
      & \\
      & \\
      & \\
      & \\
     \underline{\textbf{\@firstsupervisor}} & &  \underline{\textbf{\@secondsupervisor}} \\
     NIP. \@firstnip & & NIP. \@secondnip    \\
     
    \end{tabular}
	
	\begin{center}
		\vspace{1cm}
		Mengetahui, \\
		\vspace{0.7cm}
		
		\begin{tabular}{lll}
			Koordinator Program Studi Ilmu Komputer \\
			& \\
      		& \\
     		& \\
      		& \\
      		& \\
			\textbf{\underline{Ir. Fariani Hermin Indiyah, MT}} \\ 
			NIP. 196002111987032001
		\end{tabular}
	\end{center}

\end{singlespace}
\addcontentsline{toc}{chapter}{\agreementname}
}

%------------------------------------------------------------
%Approval Page
%------------------------------------------------------------
\def\approvalpage{%
  \chapter*{\approvalname}%
  \begin{singlespace}
\noindent 	Skripsi yang berjudul \\
    \begin{tabular}{p{0.5cm} p{13.0cm}}
      &   \@titleind \\
    \end{tabular} \\
\noindent 	disusun oleh\\
    \begin{tabular}{p{0.5cm} p{1.5cm} p{10.0cm}}
     & Nama &  :\@fullname \\
     & NIM &  :\@idnum \\
    \end{tabular} \\    
telah dipertahankan di hadapan sidang Panitia Ujian Skripsi FMIPA UNNES pada \@approvaldate
	
    \vspace{1.0cm}
    \begin{singlespace}
 \noindent  \begin{tabular}{p{7.2cm} p{7.2cm}}
      Panitia: & \\
      Ketua & Sekretaris \\
      & \\
      & \\
      & \\
      \@dean &  \@secretary \\
      NIP. \@deannip & NIP. \@secretarynip    \\
       & \\
      Penguji 1 &  \\
      & \\
      & \\
      & \\
      \@examinera  &   \\
      NIP. \@examinernipa &  \\
       & \\
      Penguji 2 &  Pembimbing \\
      & \\
      & \\
      & \\
      \@examinerb &   \@firstsupervisor \\
      NIP. \@examinernipb &  NIP. \@firstnip  \\
    \end{tabular}
    \end{singlespace}

  \end{singlespace}
\addcontentsline{toc}{chapter}{\approvalname}
}

%------------------------------------------------------------
%Acknowlegment
%------------------------------------------------------------
\def\acknowledgment{%
  \chapter*{\acknowledgmentname}%
%\null\vfill%Force the content of acknowledment to bottom page 
\addcontentsline{toc}{chapter}{\acknowledgmentname}
}

%------------------------------------------------------------
%Preface
%------------------------------------------------------------
\def\preface{%
  \chapter*{\prefacename}%
\addcontentsline{toc}{chapter}{\prefacename}
}

%------------------------------------------------------------
%Nomenclature
%------------------------------------------------------------
\def\singkatan{%
  \chapter*{\nomenclaturename}%
  \thispagestyle{plain}
  %\chapter*{\nomenclaturename}%
  %\printglossary
\addcontentsline{toc}{chapter}{\nomenclaturename}
}

%------------------------------------------------------------
%Redefine Abstract in Indonesian
%------------------------------------------------------------
\def\abstractind{%
  \clearpage\thispagestyle{chapterheading}%
  \chapter*{\abstractname}%
  %{\parindent \z@ \centering \normalfont
  %{\large\textbf{\abstractname}} \\
  %\vspace{1.0cm}
   % }
  \begin{singlespacing}%
  %\vskip 10\p@
  \addcontentsline{toc}{chapter}{\abstractname}
}
\def\endabstractind{\par\vfil\null\end{singlespacing}%
%\selectlanguage{bahasa}\clearpage\pagenumbering{arabic}\setcounter{page}{1}
}


%------------------------------------------------------------
%Redefine Abstract in English
%------------------------------------------------------------
\def\abstracteng{%
	\clearpage\thispagestyle{chapterheading}%
	\chapter*{ABSTRACT}%
	%{\parindent \z@ \centering \normalfont
		%{\large\textbf{\abstractname}} \\
		%\vspace{1.0cm}
		% }
	\begin{singlespacing}%
		%\vskip 10\p@
		\addcontentsline{toc}{chapter}{ABSTRACT}
	}
	\def\endabstracteng{\par\vfil\null\end{singlespacing}%
	%\selectlanguage{bahasa}\clearpage\pagenumbering{arabic}\setcounter{page}{1}
}


%------------------------------------------------------------
% Daftar Lampiran
%------------------------------------------------------------
%\def\daftarlampiran{%
%  \clearpage\thispagestyle{chapterheading}%
%%  \chapter*{\listattachmentsname}%

%  \begin{singlespacing}%
  
%\newcommand\appcaption[1]{%
%   \addcontentsline{app}{chapter}{#1}}
%\makeatletter
%    \newcommand\listattachmentsname{%
 %  \chapter*{\listattachmentsname}\@starttoc{app}}
%\makeatother
  
%  \vskip 20\p@
%  \listattachmentsname  
%  \end{singlespacing}
%  \addcontentsline{toc}{chapter}{\listattachmentsname}
%}


%Tabularization
\newcommand{\itab}[1]{\hspace{0em}\rlap{#1}}
\newcommand{\tab}[1]{\hspace{.2\textwidth}\rlap{#1}}


\AtBeginDocument{%
\if@doublesp%
\doublespacing%
\else\if@singlesp%
\singlespacing%
\else
\onehalfspacing%
\fi%
\fi
}

\AtBeginDocument{%
\if@langindo%
\selectlanguage{bahasa}%
\else
\selectlanguage{english}%
\fi
}

\endinput
